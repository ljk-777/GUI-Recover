from dataclasses import fields
import json

import numpy as np
from PIL import Image
import pytest

from gui_recover.actions import Action
from gui_recover.verification import (
    TransitionView,confidence_verdict,fit_classifier,observation_key,pixel_features,probabilities,
)
from gui_recover.verification_data import observation_rows
from gui_recover.verification_training import choose_threshold


def test_verifier_boundary_admits_only_visual_observations_and_agent_information():
    image=Image.new("RGB",(20,20),"white")
    view=TransitionView(image,image,Action("INPUT","word",(0.5,0.5)),"Enter word","text field")
    assert {f.name for f in fields(view)}=={"before","after","proposed","goal","target","phase"}
    with pytest.raises(TypeError):
        TransitionView(image,image,view.proposed,"goal","target",reward=1)
    with pytest.raises(ValueError,match="phase"):
        TransitionView(image,image,view.proposed,"goal","target","fault")
    assert np.isfinite(pixel_features(view)).all()
    assert observation_key(view)!=observation_key(TransitionView(image,image,view.proposed,"goal","target","after_wait"))


def test_confidence_rejection_and_invalid_probability_handling():
    assert confidence_verdict([0.1,0.8,0.1],0.9)["status"]=="unknown"
    assert confidence_verdict([0.1,0.8,0.1],0.7)["status"]=="not_achieved"
    assert confidence_verdict([0,1,0],1.01)["status"]=="unknown"
    for values in ([0.5,0.5],[-1,1,1],[float("nan"),0,1],[1,1,1]):
        with pytest.raises(ValueError):
            confidence_verdict(values,0.5)


def test_linear_head_learns_three_classes_and_reload_preserves_predictions(tmp_path):
    rng=np.random.default_rng(4)
    features=np.concatenate([rng.normal(i*3,0.1,(10,12)) for i in (-1,0,1)])
    labels=[label for label in ("achieved","unknown","not_achieved") for _ in range(10)]
    model=fit_classifier(features,labels,np.ones(30),l2=0.01,rank=4,steps=600)
    values=probabilities(model,features)
    assert np.array_equal(values.argmax(axis=1),[0]*10+[2]*10+[1]*10)
    path=tmp_path/"head.npz"
    np.savez_compressed(path,**model)
    with np.load(path,allow_pickle=False) as archive:
        reloaded={k:archive[k] for k in archive.files}
    assert np.array_equal(values,probabilities(reloaded,features))


def test_ambiguous_identical_inputs_keep_raw_truth_but_receive_unknown_label(tmp_path):
    Image.new("RGB",(20,20),"white").save(tmp_path/"blank.png")
    Image.new("RGB",(20,20),"black").save(tmp_path/"filled.png")
    records=[]
    for name,deferred in (("drop-input",False),("deferred-input",True)):
        records.append({"id":name,"scene_id":"scene","split":"train","status":"ok",
            "proposed_action":Action("INPUT","word",(0.5,0.5)).to_dict(),"goal":"Enter word","target":"text field",
            "before":{"image":"blank.png"},"immediate":{"image":"blank.png","effect_achieved":False,"terminated":False,"truncated":False},
            "after_wait":{"image":"filled.png" if deferred else "blank.png","effect_achieved":deferred,"terminated":False,"truncated":False},
            "reference_achieved":True,"regime":name,"deferred":deferred})
    views,collisions=observation_rows(tmp_path,records)
    assert len(collisions)==1
    assert [r["label"] for r in views if r["phase"]=="immediate"]==["unknown","unknown"]
    assert next(r for r in views if r["transition_id"]=="drop-input" and r["phase"]=="after_wait")["label"]=="not_achieved"


def test_calibration_guards_normal_and_pending_without_using_natural_failures():
    def record(name,regime,achieved,reference=True,deferred=False):
        state={"effect_achieved":achieved,"terminated":False,"truncated":False}
        return {"id":name,"status":"ok","regime":regime,"reference_achieved":reference,"deferred":deferred,
                "immediate":state,"after_wait":state}
    records=[record("healthy","none",True),record("fault","drop-input",False),
             record("pending","deferred-input",False,deferred=True),record("natural","none",False,reference=False)]
    views=[{"transition_id":r["id"],"phase":p} for r in records for p in ("immediate","after_wait")]
    values=np.array([[0.1,0.6,0.3],[0.98,0.01,0.01],[0.01,0.98,0.01],[0.01,0.98,0.01],
        [0.1,0.6,0.3],[0.98,0.01,0.01],[0.01,0.98,0.01],[0.01,0.98,0.01]])
    config={"thresholds":[0.5,0.9,1.01],"normal_false_trigger_budget":0.05,"premature_pending_trigger_budget":0}
    first,_=choose_threshold(records,views,values,config)
    assert first["threshold"]==0.9
    records[-1]["immediate"]["effect_achieved"]=True
    second,_=choose_threshold(records,views,values,config)
    assert first==second
