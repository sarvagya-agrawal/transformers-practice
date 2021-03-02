"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
---
Semantic Machines\N{TRADE MARK SIGN} software.

Creates the prediction report from onmt_translate output.
"""
from typing import Dict, Iterator, List, Union, Tuple, Optional

import csv

from pydantic.dataclasses import dataclass
from configargparse import Namespace
from more_itertools import chunked

import pandas as pd
import dataclasses
import jsons

from ..data.smcalflow.dialogue import (
    AgentUtterance,
    Dialogue,
    ProgramExecutionOracle,
    Turn,
    TurnId,
    UserUtterance,
)
from ..data.smcalflow.io import (
    load_jsonl_file,
    load_jsonl_file_and_build_lookup,
    save_jsonl_file,
)
from ..data.smcalflow.linearize import seq_to_lispress, to_canonical_form
from ..data.smcalflow.lispress import render_compact
from ..data.smcalflow.prediction_report import (
    PredictionReportDatum,
    save_prediction_report_tsv,
    save_prediction_report_txt,
)

_DUMMY_USER_UTTERANCE = UserUtterance(original_text="", tokens=[])
_DUMMY_AGENT_UTTERANCE = AgentUtterance(
    original_text="", tokens=[], described_entities=[]
)
_PARSE_ERROR_LISPRESS = '(parseError #(InvalidLispress "")'


@dataclass(frozen=True)
class OnmtPredictionReportDatum(PredictionReportDatum):
    datum_id: TurnId
    source: str
    # The tokenized gold lispress.
    gold: str
    # The tokenized predicted lispress.
    prediction: str
    program_execution_oracle: ProgramExecutionOracle

    @property
    def gold_canonical(self) -> str:
        return to_canonical_form(self.gold)

    @property
    def prediction_canonical(self) -> str:
        try:
            return to_canonical_form(self.prediction)
        except Exception:  # pylint: disable=W0703
            return _PARSE_ERROR_LISPRESS

    @property
    def is_correct(self) -> bool:
        return (
            self.gold == self.prediction
            and self.program_execution_oracle.refer_are_correct
        )

    @property
    def is_correct_leaderboard(self) -> bool:
        """Returns true if the gold and the prediction match after canonicalization.

        This is the metric used in the leaderboard, which would be slightly higher than the one reported in the TACL2020
        paper, since the named arguments are sorted after canonicalization.
        """
        return (
            self.gold_canonical == self.prediction_canonical
            and self.program_execution_oracle.refer_are_correct
        )

    def flatten_datum_id(self) -> Dict[str, Union[str, int]]:
        return {
            "dialogueId": self.datum_id.dialogue_id,
            "turnIndex": self.datum_id.turn_index,
        }

    def flatten(self) -> Dict[str, Union[str, int]]:
        flatten_datum_dict = self.flatten_datum_id()
        # It's fine to call update since we always return a new dict from self.flatten_datum_id().
        flatten_datum_dict.update(
            {
                "source": self.source,
                "gold": self.gold,
                "prediction": self.prediction,
                "goldCanonical": self.gold_canonical,
                "predictionCanonical": self.prediction_canonical,
                "oracleResolveAreCorrect": self.program_execution_oracle.refer_are_correct,
                "isCorrect": self.is_correct,
                "isCorrectLeaderboard": self.is_correct_leaderboard,
            }
        )
        return flatten_datum_dict


def build_prediction_report_datum(
    datum_lookup: Dict[str, Dict[int, Turn]],
    datum_id_line: str,
    src_line: str,
    ref_line: str,
    nbest_lines: List[str],
) -> OnmtPredictionReportDatum:
    datum_id = jsons.loads(datum_id_line.strip(), TurnId)
    datum = datum_lookup[datum_id.dialogue_id][datum_id.turn_index]
    return OnmtPredictionReportDatum(
        datum_id=datum_id,
        source=src_line.strip(),
        gold=ref_line.strip(),
        prediction=nbest_lines[0].strip(),
        program_execution_oracle=datum.program_execution_oracle,
    )


def create_onmt_prediction_report(
    datum_lookup: Dict[str, Dict[int, Turn]],
    datum_id_jsonl: str,
    src_txt: str,
    ref_txt: str,
    nbest_txt: str,
    nbest: int,
    outbase: str,
) -> str:
    prediction_report = [
        build_prediction_report_datum(
            datum_lookup=datum_lookup,
            datum_id_line=datum_id_line,
            src_line=src_line,
            ref_line=ref_line,
            nbest_lines=nbest_lines,
        )
        for datum_id_line, src_line, ref_line, nbest_lines in zip(
            open(datum_id_jsonl),
            open(src_txt),
            open(ref_txt),
            chunked(open(nbest_txt), nbest),
        )
    ]
    prediction_report.sort(key=lambda x: dataclasses.astuple(x.datum_id))
    predictions_jsonl = f"{outbase}.prediction_report.jsonl"
    save_jsonl_file(prediction_report, predictions_jsonl)
    save_prediction_report_tsv(
        prediction_report, f"{outbase}.prediction_report.tsv")
    save_prediction_report_txt(
        prediction_report=prediction_report,
        prediction_report_txt=f"{outbase}.prediction_report.txt",
        field_names=[
            "dialogueId",
            "turnIndex",
            "source",
            "oracleResolveAreCorrect",
            "isCorrect",
            "isCorrectLeaderboard",
            "gold",
            "prediction",
            "goldCanonical",
            "predictionCanonical",
        ],
    )
    return predictions_jsonl


def build_dataflow_dialogue(
    dialogue_id: str, prediction_report_data: Dict[int, OnmtPredictionReportDatum]
) -> Dialogue:
    turns: List[Turn] = []
    datum: OnmtPredictionReportDatum
    for turn_index, datum in sorted(prediction_report_data.items(), key=lambda x: x[0]):
        # pylint: disable=broad-except
        tokenized_lispress = datum.prediction.split(" ")
        try:
            lispress = render_compact(seq_to_lispress(tokenized_lispress))
        except Exception as e:
            print(e)
            lispress = _PARSE_ERROR_LISPRESS

        turns.append(
            Turn(
                turn_index=turn_index,
                user_utterance=_DUMMY_USER_UTTERANCE,
                agent_utterance=_DUMMY_AGENT_UTTERANCE,
                lispress=lispress,
                skip=False,
                program_execution_oracle=None,
            )
        )

    return Dialogue(dialogue_id=dialogue_id, turns=turns)


def build_dataflow_dialogues(
    prediction_report_data_lookup: Dict[str,
                                        Dict[int, OnmtPredictionReportDatum]]
) -> Iterator[Dialogue]:
    for dialogue_id, prediction_report_data in prediction_report_data_lookup.items():
        dataflow_dialogue = build_dataflow_dialogue(
            dialogue_id=dialogue_id, prediction_report_data=prediction_report_data
        )
        yield dataflow_dialogue


@dataclass
class EvaluationScores:
    num_total_turns: int = 0
    num_correct_turns: int = 0
    num_turns_before_first_error: int = 0
    num_total_dialogues: int = 0
    num_correct_dialogues: int = 0

    @property
    def accuracy(self) -> float:
        if self.num_total_turns == 0:
            return 0
        return self.num_correct_turns / self.num_total_turns

    @property
    def ave_num_turns_before_first_error(self) -> float:
        if self.num_total_dialogues == 0:
            return 0
        return self.num_turns_before_first_error / self.num_total_dialogues

    @property
    def pct_correct_dialogues(self) -> float:
        if self.num_total_dialogues == 0:
            return 0
        return self.num_correct_dialogues / self.num_total_dialogues

    def __iadd__(self, other: object) -> "EvaluationScores":
        if not isinstance(other, EvaluationScores):
            raise ValueError()
        self.num_total_turns += other.num_total_turns
        self.num_correct_turns += other.num_correct_turns
        self.num_turns_before_first_error += other.num_turns_before_first_error
        self.num_total_dialogues += other.num_total_dialogues
        self.num_correct_dialogues += other.num_correct_dialogues

        return self

    def __add__(self, other: object) -> "EvaluationScores":
        if not isinstance(other, EvaluationScores):
            raise ValueError()
        result = EvaluationScores()
        result += self
        result += other

        return result


def evaluate_dialogue(turns: List[Tuple[int, bool]]) -> EvaluationScores:
    num_correct_turns = 0
    dialogue_is_correct = True
    num_turns_before_first_error = 0
    seen_error = False
    for _turn_index, is_correct in sorted(turns, key=lambda x: x[0]):
        if is_correct:
            num_correct_turns += 1
            if not seen_error:
                num_turns_before_first_error += 1
        else:
            dialogue_is_correct = False
            seen_error = True

    return EvaluationScores(
        num_total_turns=len(turns),
        num_correct_turns=num_correct_turns,
        num_turns_before_first_error=num_turns_before_first_error,
        num_total_dialogues=1,
        num_correct_dialogues=1 if dialogue_is_correct else 0,
    )


def evaluate_dataset(
    prediction_report_df: pd.DataFrame, use_leaderboard_metric: bool
) -> EvaluationScores:
    # pylint: disable=singleton-comparison
    dataset_scores = EvaluationScores()
    if use_leaderboard_metric:
        field_name = "isCorrectLeaderboard"
    else:
        field_name = "isCorrect"
    for _dialogue_id, df_for_dialogue in prediction_report_df.groupby("dialogueId"):
        turns = [
            (int(row.get("turnIndex")), row.get(field_name))
            for _, row in df_for_dialogue.iterrows()
        ]
        dialogue_scores = evaluate_dialogue(turns)
        dataset_scores += dialogue_scores

    return dataset_scores


def main(args: Namespace) -> None:
    dialogues_jsonl = args.data
    datum_id_jsonl = args.datum_ids
    src_txt = args.src
    ref_txt = args.ref
    nbest_txt = args.nbest_txt
    nbest = args.nbest
    outbase = args.output
    """Creates 1-best predictions and saves them to files."""
    datum_lookup: Dict[str, Dict[int, Turn]] = {
        dialogue.dialogue_id: {
            turn.turn_index: turn for turn in dialogue.turns}
        for dialogue in load_jsonl_file(
            data_jsonl=dialogues_jsonl, cls=Dialogue, unit=" dialogues"
        )
    }

    prediction_report_jsonl = create_onmt_prediction_report(
        datum_lookup=datum_lookup,
        datum_id_jsonl=datum_id_jsonl,
        src_txt=src_txt,
        ref_txt=ref_txt,
        nbest_txt=nbest_txt,
        nbest=nbest,
        outbase=outbase,
    )

    predictions_lookup = load_jsonl_file_and_build_lookup(
        data_jsonl=prediction_report_jsonl,
        cls=OnmtPredictionReportDatum,
        primary_key_getter=lambda x: x.datum_id.dialogue_id,
        secondary_key_getter=lambda x: x.datum_id.turn_index,
    )
    dataflow_dialogues = build_dataflow_dialogues(predictions_lookup)
    save_jsonl_file(dataflow_dialogues, f"{outbase}.dataflow_dialogues.jsonl")

    prediction_report_tsv = f"{outbase}.prediction_report.tsv"
    use_leaderboard_metric = args.leaderboard
    scores_json = args.scores_json
    prediction_report_df = pd.read_csv(
        prediction_report_tsv,
        sep="\t",
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        na_values=None,
        keep_default_na=False,
    )
    assert not prediction_report_df.isnull().any().any()

    # if datum_ids_jsonl:
    #     datum_ids = set(
    #         load_jsonl_file(data_jsonl=datum_ids_jsonl,
    #                         cls=TurnId, verbose=False)
    #     )
    #     mask_datum_id = [
    #         TurnId(dialogue_id=row.get("dialogueId"),
    #                turn_index=row.get("turnIndex"))
    #         in datum_ids
    #         for _, row in prediction_report_df.iterrows()
    #     ]
    #     prediction_report_df = prediction_report_df.loc[mask_datum_id]

    scores = evaluate_dataset(prediction_report_df, use_leaderboard_metric)
    with open(scores_json, "w") as fp:
        fp.write(jsons.dumps(scores, jdkwargs={"indent": 2}))
        fp.write("\n")
