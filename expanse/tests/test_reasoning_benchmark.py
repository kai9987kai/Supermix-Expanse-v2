"""The diagnostic must not turn memorized numbers or consistent errors into wins."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import evaluate_reasoning as evaluator
import reasoning_benchmark as rb


@pytest.mark.parametrize("reply", ["12", "12.0", "1.2e1", "the answer is 12.",
                                   "3 * 4 = 12, total 12", "x is 12, total 12"])
def test_explicit_terminal_numbers(reply):
    assert rb.score_reply(reply, 12)["correct"]


@pytest.mark.parametrize("reply", ["120", "112", "12 or 13", "12\n13", "total 12 or 13",
                                   "total 12, final answer 13", "not 12, total 12",
                                   "the answer is not 12", "maybe the answer is 12",
                                   "wrong answer 12, total 12", "total 12 because 12 is nice",
                                   "I saw 12 earlier", "3 * 4 = 12", "NaN", "inf", ""])
def test_ambiguous_or_incidental_numbers_do_not_pass(reply):
    assert not rb.score_reply(reply, 12)["correct"]


def test_same_value_repeated_markers_and_wrong_terminal_answer():
    assert rb.score_reply("answer 12.0, total 12", 12)["correct"]
    assert not rb.score_reply("12 is an operand, total 13", 12)["correct"]
    assert rb.score_reply("total -2", -2)["correct"]


@pytest.mark.parametrize("reply", ["6 x 7 = 42, total 42", "6 x 7 = 42, the answer is 42",
                                   "answer:42", "answer = 42", "x=42", "x is 42"])
def test_multiplication_x_is_not_an_answer_marker(reply):
    assert rb.score_reply(reply, 42)["correct"]


@pytest.mark.parametrize("reply", ["answer42", "total42", "x42", "x 42", "final answer42"])
def test_answer_markers_require_word_boundaries_and_separators(reply):
    assert not rb.score_reply(reply, 42)["valid_answer"]


def test_intermediate_x_states_remain_conservatively_ambiguous():
    score = rb.score_reply("x = 6, x = 42, total 42", 42)
    assert not score["valid_answer"]
    assert score["parse_reason"] == "conflicting_answer_markers"


def test_compound_expressions_are_not_mislabeled_as_binary_semantics():
    assert rb.semantic_signature("What is 2 + 3 * 4?") is None
    assert rb.semantic_signature("What is 3 + 2?") == rb.semantic_signature("Solve: 2 + 3")
    assert rb.semantic_signature("What is 3 - 2?") != rb.semantic_signature("What is 2 - 3?")


def test_suite_deterministic_balanced_complete_and_solver_correct(tmp_path):
    first = rb.make_suite([], seed=9, groups_per_family=3)
    assert first == rb.make_suite([], seed=9, groups_per_family=3)
    rb.validate_suite(first)
    assert len(first["groups"]) == 6
    for group in first["groups"]:
        spec = group["spec"]
        for item in group["items"]:
            if spec["family"] == "python_trace":
                # Independent bounded interpreter for the emitted literal loop.
                lines = item["user"].splitlines()
                start = next(line for line in lines if line.startswith("x = "))
                loop = next(line for line in lines if line.startswith("for i in range("))
                x = int(start.split("=")[1])
                count = int(loop.split("range(")[1].split(")")[0])
                delta = int(loop.rsplit(" ", 1)[1])
                for _ in range(count):
                    x += -delta if "x -" in loop else delta
                assert item["expected"] == x
            else:
                signature = rb.semantic_signature(item["user"])
                _, a, op, b = signature
                expected = {"+": lambda: a + b, "-": lambda: a - b,
                            "*": lambda: a * b, "/": lambda: a // b}[op]()
                assert item["expected"] == expected
    path = tmp_path / "suite.json"
    evaluator.write_new_json(path, first)
    assert rb.load_suite(path) == first
    evaluator.write_new_json(path, first)
    with pytest.raises(FileExistsError):
        evaluator.write_new_json(path, rb.make_suite([], seed=10))


def test_semantic_overlap_excludes_whole_group_and_binds_corpus_bytes(tmp_path):
    original = rb.make_suite([], seed=8, groups_per_family=1)
    hit = original["groups"][0]
    question = "A different wrapper: " + hit["items"][0]["user"]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"user": question, "assistant": "irrelevant"}) + "\n", encoding="utf-8")
    screened = rb.make_suite([corpus], seed=8, groups_per_family=1)
    assert hit["id"] not in {g["id"] for g in screened["groups"]}
    assert screened["corpus_audit"]["excluded_groups"]["recognized_semantic_group_overlap"] >= 1
    assert screened["corpus_audit"]["files"][0]["recognized_semantic_rows"] == 1
    rb.verify_corpus(screened)
    corpus.write_text(json.dumps({"user": question, "assistant": "changed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus bytes changed"):
        rb.verify_corpus(screened)


def test_suite_hash_and_solver_labels_both_enforced():
    suite = rb.make_suite([], seed=4, groups_per_family=1)
    suite["groups"][0]["items"][0]["expected"] += 1
    with pytest.raises(ValueError, match="SHA-256"):
        rb.validate_suite(suite)
    suite["sha256"] = rb.canonical_sha256({k: v for k, v in suite.items() if k != "sha256"})
    with pytest.raises(ValueError, match="solver-labelled"):
        rb.validate_suite(suite)


def test_fixed_selection_preserves_groups_and_both_families():
    suite = rb.make_suite([], seed=4, groups_per_family=3)
    subset = rb.select_groups(suite, 3)
    assert len(subset) == 3
    assert all(len(g["items"]) == 6 for g in subset)
    assert {g["spec"]["family"] for g in subset} == {"arithmetic", "python_trace"}
    assert subset == rb.select_groups(suite, 6)[:3]
    with pytest.raises(ValueError):
        rb.select_groups(suite, 1)


def results_for(groups, mode):
    results = []
    for group in groups:
        for item in group["items"]:
            reply = str(item["expected"]) if mode == "correct" else ("total 9999999" if mode == "constant" else "no answer")
            results.append({"id": item["id"], **rb.score_reply(reply, item["expected"])})
    return results


def test_constant_wrong_outputs_do_not_pass_consistency_or_contrast_accuracy():
    groups = rb.select_groups(rb.make_suite([], groups_per_family=1))
    metrics = rb.summarize(groups, results_for(groups, "constant"))
    assert metrics["paraphrase_consistency"] == 1
    assert metrics["accuracy"] == metrics["family_macro_accuracy"] == 0
    assert metrics["all_correct_group_rate"] == metrics["contrast_pair_accuracy"] == 0
    assert metrics["contrast_changed_answer_rate"] == 0


def test_all_failures_stay_in_denominator_and_missing_results_rejected():
    groups = rb.select_groups(rb.make_suite([], groups_per_family=1))
    rows = results_for(groups, "invalid")
    metrics = rb.summarize(groups, rows)
    assert metrics["n"] == metrics["invalid_answers"] == 12
    assert metrics["accuracy"] == 0
    with pytest.raises(ValueError, match="every selected item"):
        rb.summarize(groups, rows[:-1])
    with pytest.raises(ValueError, match="every selected item"):
        rb.summarize(groups, rows + rows[:1])


def test_perfect_answers_pass_every_correctness_metric():
    groups = rb.select_groups(rb.make_suite([], groups_per_family=1))
    metrics = rb.summarize(groups, results_for(groups, "correct"))
    for key in ("accuracy", "family_macro_accuracy", "all_correct_group_rate", "paraphrase_pair_accuracy",
                "paraphrase_consistency", "contrast_pair_accuracy", "contrast_changed_answer_rate"):
        assert metrics[key] == 1


def test_context_budget_never_truncates_prompt_or_exceeds_context():
    assert evaluator.generation_budget(40, 96, 128) == 88
    assert evaluator.generation_budget(40, 64, 128) == 64
    with pytest.raises(ValueError, match="no answer budget"):
        evaluator.generation_budget(128, 64, 128)
    with pytest.raises(ValueError):
        evaluator.generation_budget(10, 0, 128)
