"""Exercise the Nash loader -> prose parser -> exact verifier reward boundary."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import CharTok, OPEN, CLOSE, END
from rlaif.structured import GOLD_PREFIX, check, policy_problem, validate_training_config, encode_gold
from rlaif.data import prepare_data
from rlaif.parsing import parse_completion
from rlaif.verify import Verifier, _gold_ok, _verify_one

ROOT=Path(__file__).resolve().parents[1]


def fixture_row():
    return json.loads((ROOT/'data/nash/rlaif.jsonl').read_text().splitlines()[0])


def test_rollout_parser_and_spawned_verifier():
    row=fixture_row()
    gold=GOLD_PREFIX+json.dumps(row['expected_answer'])
    reply='The quadratic welfare identity checks the loss.\nResult JSON:\n'+json.dumps(row['expected_answer'])
    tok=CharTok()
    completion=OPEN+'Before computing, I compare the marginal conditions.\n\nGeneral pattern: Cancel transfers before measuring welfare.'+CLOSE+reply+END
    parsed=parse_completion(tok.encode(completion),tok,think_open=OPEN,think_close=CLOSE,
                            opened_in_prompt=False,finish_reason='stop',max_steps=24,
                            answer_mode='plain',require_box=False,reasoning_format='prose')
    assert parsed.hard_valid and parsed.format_valid
    assert parsed.final_answer==reply
    verifier=Verifier(workers=1,timeout_s=20)
    try:
        assert verifier.gold_ok([gold])==[True]
        wrong=copy.deepcopy(row['expected_answer']);wrong['outputs']=[99,99,99]
        assert verifier.check([(gold,parsed.final_answer),(gold,json.dumps(wrong))])==[('ok',True),('ok',False)]
    finally:
        verifier.close()


def test_strict_payload_boundary_and_operational_distinction():
    gold=GOLD_PREFIX+json.dumps({'q':'1/2','active':True})
    assert _verify_one(gold,'{"q":"2/4","active":true}')==('ok',True)
    for bad in ['{"q":"1/2","active":1}',
                '{"q":99,"q":"1/2","active":true}',
                '{"q":"1/2","active":true} trailing',
                'Result JSON:\n{}\nResult JSON:\n{"q":"1/2","active":true}',
                '{"q":"__import__(\"os\")","active":true}',
                '{"q":NaN,"active":true}']:
        assert check(gold,bad)==('ok',False)
    assert _verify_one(GOLD_PREFIX+'{}','{}')==('gold_unparseable',False)
    assert not _gold_ok(GOLD_PREFIX+'{}')


def test_loader_preserves_splits_and_excludes_hint_rows():
    dc=SimpleNamespace(sources=['nash'],nash_path=str(ROOT/'data/nash/rlaif.jsonl'),
                       answer_format='plain',reasoning_format='prose')
    verifier=SimpleNamespace(gold_ok=lambda golds:[_gold_ok(g) for g in golds])
    loaded=prepare_data(dc,0,verifier)
    exported=[json.loads(line) for line in Path(dc.nash_path).read_text().splitlines()]
    all_loaded={p.id:p for split in loaded.values() for p in split}
    assert set(all_loaded)=={r['id'] for r in exported}
    hints={'nash:0086','nash:0130','nash:0243','nash:0306'}
    assert not hints & set(all_loaded)
    for row in exported:
        item=all_loaded[row['id']]
        assert item.split==row['split'] and item in loaded[row['split']]
        spec=json.loads(item.gold[len(GOLD_PREFIX):])
        assert spec['expected']==row['expected_answer']
        assert spec['symbol_assumptions']==row.get('symbol_assumptions',{})
        assert item.problem==policy_problem(row)
        assert item.source=='nash' and len(item.source_revision)==64
    dc.answer_format='boxed'
    with pytest.raises(ValueError,match='answer_format'):
        prepare_data(dc,0,verifier)
    dc.sources=['nash','gsm8k']
    with pytest.raises(ValueError,match='sources'):
        prepare_data(dc,0,verifier)


def test_policy_schema_does_not_contain_reference_values():
    row={'problem':'Find the result.','expected_answer':{'q':'987654321/111111111','active':False}}
    prompt=policy_problem(row)
    assert '987654321' not in prompt and '111111111' not in prompt
    assert '"q": "exact expression"' in prompt and '"active": "boolean"' in prompt


def test_training_preflight_preserves_exact_reward_and_four_k_context():
    cfg=SimpleNamespace(data=SimpleNamespace(sources=['nash']),
                        judge=SimpleNamespace(backend='qwen'),
                        reward_v2=SimpleNamespace(enabled=False),
                        reward_v3=SimpleNamespace(enabled=False),
                        train=SimpleNamespace(max_prompt_tokens=1280,max_completion_tokens=2816))
    validate_training_config(cfg)
    cfg.train.max_completion_tokens=2817
    with pytest.raises(ValueError,match='4096'):
        validate_training_config(cfg)
    cfg.train.max_completion_tokens=2816
    cfg.judge.backend='deepseek'
    with pytest.raises(ValueError,match='reward_v3'):
        validate_training_config(cfg)
    cfg.reward_v3.enabled=True                      # the reward service grades the exact outcome itself
    validate_training_config(cfg)
    cfg.train.max_completion_tokens=2817            # the 4K cap holds with the service too
    with pytest.raises(ValueError,match='4096'):
        validate_training_config(cfg)
    cfg.train.max_completion_tokens=2816
    cfg.judge.backend='qwen';cfg.reward_v2.enabled=True
    with pytest.raises(ValueError,match='reward_v2'):
        validate_training_config(cfg)


def test_trusted_domains_accept_rationalized_root_without_new_singularities():
    expected={'root':'(sqrt(9*n**2+32*v)-3*n)/8'}
    candidate='{"root":"4*v/(sqrt(9*n**2+32*v)+3*n)"}'
    domains={'n':{'nonnegative':True},'v':{'positive':True}}
    gold=encode_gold(expected,domains)
    assert _verify_one(gold,candidate)==('ok',True)
    assert _verify_one(encode_gold({'x':1},{'n':{'nonnegative':True}}),'{"x":"n/n"}')==('ok',False)
    assert _verify_one(encode_gold({'x':1},domains),'{"x":"(n-3)/(n-3)"}')==('ok',False)
    # Candidate text cannot supply or widen the trusted domains.
    assert _verify_one(encode_gold({'x':1},domains),'{"x":"1","symbol_assumptions":{"n":{"positive":true}}}')==('ok',False)
    assert not _gold_ok(encode_gold(expected,{'n':{'integer':'yes'}}))
