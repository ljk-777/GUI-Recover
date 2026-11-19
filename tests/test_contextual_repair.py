from dataclasses import fields
import json

from PIL import Image
import pytest

from gui_recover.actions import Action
from gui_recover.contextual_repair import ContextualRepairPolicy, RepairGrounder, prepare_repair
from gui_recover.verification import TransitionView, pixel_hash
from gui_recover.repair_study import select_repair


def view(point=(0.96, 0.03)):
    image = Image.new("RGB", (500, 320), "white")
    image.putpixel((490, 10), (0, 0, 0))
    return TransitionView(image, image.copy(), Action("CLICK", position=point), "Click Tab 3", "Tab #3")


def test_plain_input_is_unchanged_and_feedback_omits_coordinates_and_labels():
    v = view()
    plain = prepare_repair(v, "plain")
    assert plain.query == v.target and pixel_hash(plain.picture) == pixel_hash(v.after)
    assert plain.map_point((0.2, 0.3)) == (0.2, 0.3)
    feedback = prepare_repair(v, "feedback")
    assert feedback.query.endswith(v.target) and '"action":"CLICK"' in feedback.query
    assert "0.96" not in feedback.query and "position" not in feedback.query
    assert {f.name for f in fields(v)} == {"before", "after", "proposed", "goal", "target", "phase"}
    with pytest.raises(TypeError):
        TransitionView(v.before, v.after, v.proposed, v.goal, v.target, target_boxes=[])


@pytest.mark.parametrize("point", [(0, 0), (1, 1), (0.99, 0.01), (0.1, 0.95), (0.5, 0.5)])
def test_zoom_keeps_canvas_inside_image_and_maps_local_coordinates_to_original(point):
    v = view(point)
    native = v.after.tobytes()
    prepared = prepare_repair(v, "zoom")
    assert prepared.picture.size == v.after.size
    left, top, right, bottom = prepared.crop
    assert right-left == 300 and bottom-top == 192
    assert 0 <= left < right <= 500 and 0 <= top < bottom <= 320
    assert prepared.map_point((0, 0)) == (left/500, top/320)
    assert prepared.map_point((1, 1)) == (right/500, bottom/320)
    assert v.after.tobytes() == native


def test_zoom_cache_maps_prediction_without_extra_model_calls_and_rejects_conflicting_prime(tmp_path):
    class Actor:
        calls = 0
        last_metrics = {}
        def ground(self, picture, query):
            self.calls += 1
            self.last_metrics = {"input_tokens": 321, "generated_tokens": 8}
            return "[0.5, 0.5]"
    actor = Actor()
    cache = RepairGrounder(actor, tmp_path)
    v = view()
    policy = ContextualRepairPolicy(cache, "feedback_zoom")
    result = policy.predict(v)
    assert tuple(result['point']) == prepare_repair(v, 'feedback_zoom').map_point((0.5, 0.5))
    assert result['local_point'] == (0.5, 0.5)
    assert policy.predict(v)['cache_hit'] is True and actor.calls == 1
    assert cache.counters()['actual_input_tokens'] == 321
    assert cache.counters()['logical_requests'] == 2
    cache.prime(v.after, v.target, '[0.1,0.2]', 'source')
    assert cache.ground(v.after, v.target)['source_prediction_provenance'] == 'source'
    with pytest.raises(ValueError, match='Conflicting'):
        cache.prime(v.after, v.target, '[0.2,0.3]', 'another')


def test_fixed_zoom_keeps_the_matched_prompt_and_ignores_the_proposed_location():
    given = view((0.96, 0.03))
    fixed = prepare_repair(given, "feedback_zoom", fixed_crop=(0, 0, .6, .6))
    other = prepare_repair(view((0.1, 0.95)), "feedback_zoom", fixed_crop=(0, 0, .6, .6))
    adaptive = prepare_repair(given, "feedback_zoom")
    assert fixed.query == other.query == adaptive.query
    assert (fixed.crop, pixel_hash(fixed.picture)) == (other.crop, pixel_hash(other.picture))
    assert fixed.crop == (0, 0, 300, 192) and adaptive.crop != fixed.crop
    assert fixed.map_point((0.5, 0.5)) == (0.3, 0.3)
    with pytest.raises(ValueError, match="normalized"):
        prepare_repair(given, "plain", fixed_crop=(0, 0, .6, .6))


def test_selection_requires_actual_recovery_gain_and_both_normal_guards():
    modes = ('plain', 'feedback', 'zoom', 'feedback_zoom')
    def row(kind, values):
        return {'split':'validation','kind':kind,'task':'task','heads':{
            mode:{'hit':hit,'point':(0.5,0.5)} for mode,hit in zip(modes,values)}}
    predictions=[row('healthy_recheck',[True,True,False,True]),
                 row('normal_submit',[True,False,True,True]),row('natural_error',[False,True,True,True])]
    scores={'oracle_'+m:{'successes':int(m!='plain')} for m in modes}
    config={'candidate_order':['feedback','zoom','feedback_zoom']}
    result=select_repair(predictions,scores,config)
    assert result['selected_mode']=='feedback_zoom' and result['eligible_modes']==['feedback_zoom']
    scores['oracle_feedback_zoom']['successes']=0
    assert select_repair(predictions,scores,config)['selected_mode']=='plain'
    predictions[0]['split']='test'
    with pytest.raises(ValueError,match='validation predictions only'):
        select_repair(predictions,scores,config)


def test_preflight_only_uses_initial_pixels_and_keeps_reserved_order(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import gui_recover.repair_study_data as data
    reference = tmp_path/'reference'; reference.mkdir()
    (reference/'train.jsonl').write_text('original-source')
    old_image=Image.new('RGB',(20,20),(1,0,0))
    monkeypatch.setattr(data,'load_transitions',lambda *_:[{
        'regime':'none','task':'click-test','seed':1,'before':{'sha256':data.png_sha(old_image)}}])
    class Execution:
        def __init__(self,task,seed,proposed):
            # 3 repeats the newly accepted seed 2, so both old/new overlaps are rejected.
            self.obs=SimpleNamespace(screenshot=Image.new('RGB',(20,20),(2 if seed==3 else seed,0,0)))
        @property
        def reward(self): raise AssertionError('Outcome read during preflight')
        def first(self): raise AssertionError('Action executed during preflight')
        def close(self): pass
    monkeypatch.setattr(data,'TransitionExecution',Execution)
    out=tmp_path/'preflight';out.mkdir()
    result=data.preflight([reference],out,[{'task':'click-test','count':2,'candidate_seeds':[1,2,3,4]}])
    assert [r['seed'] for r in result['selected']]==[2,4]
    assert [r['seed'] for r in result['rejected_layouts']]==[1,3]
    assert result['actor_calls']==0 and result['outcomes_observed'] is False
