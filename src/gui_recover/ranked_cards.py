"""Inspect ranked visual search results in order within the prediction budget."""

import json

from .actions import Action, parse_action
from .listing_cards import RESULT_VIEW_HEIGHT, card_panel, inspect_card_receipt, read_card_checks, visible_result_range
from .qwen_grounding import ask_observation
from .verification import pixel_hash


BATCH_SIZE = 6
ARTWORK_QUERIES = {"painting", "paintings", "drawing", "drawings", "illustration", "artwork"}


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
        self.verified = 0
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

        subject = "an animal" if criterion.casefold() == "animals" else criterion

        def ask(items, prompt):
            panel = card_panel(items)
            raw = ask_observation(actor, actor.prepare_image(panel), prompt + (
                f'Return only one JSON object with keys "1" through "{len(items)}". Each value must be '
                "[true or false, short visual cue], with at most three words for its cue. Use null when the cue is unclear."
            ), observation)
            metrics.update(model_invocations=1, provider_output=raw, ranked_panel_pixels=pixel_hash(panel),
                           peak_cuda_allocated_bytes=actor.last_metrics["peak_cuda_allocated_bytes"])
            try:
                return read_card_checks(raw, len(items))
            except (ValueError, RecursionError):
                return None

        if len(self.selected) == count:
            item = self.selected[self.verified]
            results = ask([item], (
                f"Inspect the single photo. Does the pictured subject visibly include {subject}? "
                "A subject portrayed in a painting, drawing or photo counts. Judge the pictured subject using "
                "a specific visible feature. Do not decide anything about price, date, relevance or result order. "
            ))
            if results != [True]:
                return finish("ranked_confirmation_withheld", stop=True)
            self.verified += 1
            if self.verified < count:
                return finish("ranked_card_confirmed")
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
        predicate = f"visibly depicts {subject}"
        if query.casefold() in ARTWORK_QUERIES:
            predicate = f"is an artwork that {predicate}"
        results = ask(items, (
            f"Task: {observation.goal}\nFor each numbered card, judge if the item {predicate}. "
            "Look at the image first and use the rendered title for context. Inspect each card independently. "
        ))
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
