"""Inspect ranked visual search results in order within the prediction budget."""

import json

from .actions import Action, parse_action
from .listing_cards import RESULT_VIEW_HEIGHT, card_panel, inspect_card_receipt, read_card_checks, visible_result_range
from .verification import pixel_hash


BATCH_SIZE = 6


class RankedCardEvidence:
    """Process complete result pages and confirm the earliest qualifying prefix."""

    def __init__(self, site):
        self.site = site
        self.expanded = False
        self.previous_last = 0
        self.total = None
        self.offset = 0
        self.page_frame = None
        self.selected = []
        self.seen = set()

    def decide(self, observation, request, actor, screen_reader, capture):
        query, criterion, count = request
        screen = observation.screenshot
        frame = pixel_hash(screen)
        metrics = {"evidence_stage": "ranked_cards", "evidence_frame": frame,
                   "provider_output": None, "model_invocations": 0, "peak_cuda_allocated_bytes": 0}

        def finish(status, action=None, *, stop=False):
            metrics.update(provider_format=status)
            if stop:
                metrics["stop_after_evidence"] = True
            if metrics["model_invocations"]:
                actor.last_metrics.update(metrics)
            else:
                actor.last_metrics = metrics
            return json.dumps(action.to_dict()) if action is not None else None

        def ask(items, stage):
            panel = card_panel(items)
            instruction = ("Independently recheck these proposed results before any submission. " if stage == "confirm"
                           else "Judge every numbered card in the order shown. ")
            raw = actor.ask(actor.prepare_image(panel), (
                f"Task: {observation.goal}\n{instruction}"
                f'For each numbered card, judge if the item fits the search "{query}" and visibly features {criterion}. '
                "Look at the image first and use the rendered title for context. Inspect each card independently. "
                f'Return only one JSON object with keys "1" through "{len(items)}", one per card. '
                "Each value is [true or false, short visual cue] with at most three words for the cue. "
                "Use null for the whole value when you cannot decide."
            ))
            metrics.update(model_invocations=1, provider_output=raw, ranked_panel_pixels=pixel_hash(panel),
                           peak_cuda_allocated_bytes=actor.last_metrics["peak_cuda_allocated_bytes"])
            try:
                return read_card_checks(raw, len(items))
            except (ValueError, RecursionError):
                return None

        if len(self.selected) == count:
            ranked = self.selected[::-1]
            results = ask(ranked, "confirm")
            if results is None or not all(value is True for value in results):
                return finish("ranked_confirmation_withheld", stop=True)
            answer = Action("ANSWER", "\n".join(card["destination"] for _, card in self.selected))
            return finish("ranked_confirmed", parse_action(json.dumps(answer.to_dict())))

        # Keep source frame and scope evidence available for offline reconstruction on each batch.
        size = actor.prepare_image(screen).size
        rows = screen_reader(screen, size)
        receipt = capture()
        metrics.update(ranked_receipt=receipt, ranked_screen_text=rows, ranked_canvas_size=list(size))
        cards = inspect_card_receipt(receipt, frame, screen.size, self.site)
        scope = visible_result_range(rows, query)
        if scope is None or receipt["scroll_top"] != 0:
            return finish("ranked_scope_missing", stop=True)
        first, last, total = scope

        if not self.expanded:
            grid = receipt["grid"]
            if first != 1 or grid is None:
                return finish("ranked_scope_missing", stop=True)
            self.expanded = True
            return finish("ranked_expand", Action("LISTING_GRID", position=tuple(grid)))

        if (screen.height != RESULT_VIEW_HEIGHT or first != self.previous_last + 1
                or (self.total is not None and total != self.total)
                or len(cards) != last - first + 1 or any(card["image"] is None for card in cards)):
            return finish("ranked_incomplete_page", stop=True)
        if self.offset == 0:
            self.total = total
            self.page_frame = frame
            if any(card["destination"] in self.seen for card in cards):
                return finish("ranked_incomplete_page", stop=True)
            self.seen.update(card["destination"] for card in cards)
        elif frame != self.page_frame:
            return finish("ranked_incomplete_page", stop=True)

        batch = cards[self.offset:self.offset + BATCH_SIZE]
        self.offset += len(batch)
        items = [(screen, card) for card in batch]
        results = ask(items, "scan")
        if results is None:
            return finish("ranked_unverified_prefix", stop=True)
        for item, matches in zip(items, results, strict=True):
            if matches is None:
                return finish("ranked_unverified_prefix", stop=True)
            if matches:
                self.selected.append(item)
                if len(self.selected) == count:
                    return finish("ranked_prefix_found")
        if self.offset < len(cards):
            return finish("ranked_batch")
        if receipt["next"] is None or last >= total:
            return finish("ranked_incomplete_page", stop=True)
        self.previous_last = last
        self.offset = 0
        return finish("ranked_next_page", Action("CLICK", position=tuple(receipt["next"])))
