from PIL import Image, ImageDraw, ImageFont

from gui_recover.actions import Action
from gui_recover.target_state import CheckboxStateVerifier
from gui_recover.verification import TransitionView


def verifier():
    policy = CheckboxStateVerifier.__new__(CheckboxStateVerifier)
    policy.font = ImageFont.load_default(size=15)
    policy.minimum_score, policy.minimum_margin = .70, .18
    return policy


def form(font, checked=None, labels=("Ab2", "WX4")):
    image = Image.new("RGB", (498, 321), "#555555")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 239, 314), fill="white")
    draw.rectangle((0, 0, 239, 74), fill="yellow")
    for i, (top, label) in enumerate(zip((83, 112), labels)):
        if checked == i:
            draw.rectangle((14, top, 33, top+18), fill=(0, 92, 200))
            draw.line([(17, top+9), (22, top+14), (31, top+4)], fill="white", width=3)
        else:
            draw.rectangle((14, top, 33, top+18), outline="#777777", width=2)
        draw.text((43, top+4), label, fill="black", font=font)
    return image


def view(before, after, point=(.048, .38)):
    return TransitionView(before, after, Action("CLICK", position=point),
                          "Select Ab2 and click Submit.", 'The checkbox labeled "Ab2"')


def test_recovery_rechecks_wrong_checkbox_before_clicking_target():
    policy = verifier()
    before = form(policy.font)
    wrong = form(policy.font, 1)
    right = form(policy.font, 0)
    touched = []

    def execute(action):
        touched.append(action.position)
        return before if len(touched) == 1 else right

    result = policy.recover(view(before, wrong), execute)
    assert [event["kind"] for event in result["actions"]] == ["undo_wrong_check", "check_target"]
    assert result["final"]["status"] == "achieved"
    assert touched[0][1] > touched[1][1]

    result = policy.recover(view(before, wrong), lambda action: wrong)
    assert [event["kind"] for event in result["actions"]] == ["undo_wrong_check"]
    assert result["final"]["reason"] == "different_checkbox_checked"


def test_ambiguous_labels_and_changes_outside_checkbox_area_abstain():
    policy = verifier()
    duplicated = form(policy.font, labels=("Ab2", "Ab2"))
    assert policy.verify(view(duplicated, duplicated))["reason"] == "ambiguous_target_label"
    after = duplicated.copy()
    after.putpixel((255, 210), (0, 0, 0))
    assert policy.verify(view(duplicated, after))["reason"] == "changes_outside_checkbox_controls"
