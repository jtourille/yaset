from seqeval.metrics import precision_score, recall_score, f1_score


def eval_ner(eval_payload: list = None):

    pred = list()
    gs = list()

    for item in eval_payload:
        pred.extend(item.get("pred"))
        gs.extend(item.get("gs"))

    payload = dict()

    payload["step"] = {
        "precision": precision_score(gs, pred),
        "recall": recall_score(gs, pred),
        "f1_score": f1_score(gs, pred),
    }

    return payload

    # all_payloads = defaultdict(list)
    #
    # for batch_payload in eval_payload:
    #     all_payloads["pred"].extend(batch_payload["pred"])
    #     all_payloads["gs"].extend(batch_payload["gs"])
    #
    # metric_payload = {"main": None, "tensorboard": list()}
    #
    # main_scores_items = list()
    #
    # # NER
    # # ---------------------------------------------------------------
    #
    # ner_overall, ner_by_type = evaluate_ner(
    #     all_payloads["gs"], all_payloads["pred"]
    # )
    #
    # metric_payload["tensorboard"].append(("ner/all_f1", ner_overall.fscore))
    # metric_payload["tensorboard"].append(("ner/all_p", ner_overall.prec))
    # metric_payload["tensorboard"].append(("ner/all_r", ner_overall.rec))
    #
    # main_scores_items.append(ner_overall.fscore)
    #
    # for name, metric in ner_by_type.items():
    #     metric_payload["tensorboard"].append(
    #         ("ner/{}_f1".format(name), metric.fscore)
    #     )
    #     metric_payload["tensorboard"].append(
    #         ("ner/{}_p".format(name), metric.prec)
    #     )
    #     metric_payload["tensorboard"].append(
    #         ("ner/{}_r".format(name), metric.rec)
    #     )
    #
    # metric_payload["main"] = ner_overall.fscore

    # return metric_payload
