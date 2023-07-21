import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
import numpy as np
from task import pubmed_task, nicta_task, dri_task, art_task

from utils import tensor_dict_to_gpu, tensor_dict_to_cpu, get_device
from eval_run import get_task


def calc_classification_metrics(y_true, y_predicted, labels):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='micro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='weighted')
    per_label_precision, per_label_recall, per_label_f1, _ = precision_recall_fscore_support(y_true, y_predicted, average=None, labels=labels)

    acc = accuracy_score(y_true, y_predicted)

    class_report = classification_report(y_true, y_predicted, digits=4)
    confusion_abs = confusion_matrix(y_true, y_predicted, labels=labels)
    # normalize confusion matrix
    confusion = np.around(confusion_abs.astype('float') / confusion_abs.sum(axis=1)[:, np.newaxis] * 100, 2)
    return {"acc": acc,
            "macro-f1": macro_f1,
            "macro-precision": macro_precision,
            "macro-recall": macro_recall,
            "micro-f1": micro_f1,
            "micro-precision": micro_precision,
            "micro-recall": micro_recall,
            "weighted-f1": weighted_f1,
            "weighted-precision": weighted_precision,
            "weighted-recall": weighted_recall,
            "labels": labels,
            "per-label-f1": per_label_f1.tolist(),
            "per-label-precision": per_label_precision.tolist(),
            "per-label-recall": per_label_recall.tolist(),
            "confusion_abs": confusion_abs.tolist()
            }, \
           confusion.tolist(), \
           class_report


def create_task(create_func):
    return create_func(train_batch_size=32, max_docs=-1)


def get_all_tasks():
    tasks = []
    tasks.append(create_task(pubmed_task))
    tasks.append(create_task(nicta_task))
    tasks.append(create_task(dri_task))
    tasks.append(create_task(art_task))
    return tasks


def eval_model(model, eval_batches, device, task):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in eval_batches:
            # move tensor to gpu
            tensor_dict_to_gpu(batch, device)

            if batch["task"] != task.task_name:
                continue

            output = model(batch=batch)

            true_labels_batch, predicted_labels_batch = \
                clear_and_map_padded_values(batch["label_ids"].view(-1), output["predicted_label"].view(-1), task.labels)

            true_labels.extend(true_labels_batch)
            predicted_labels.extend(predicted_labels_batch)

            tensor_dict_to_cpu(batch)

    metrics, confusion, class_report = \
        calc_classification_metrics(y_true=true_labels, y_predicted=predicted_labels, labels=task.labels)
    return metrics, confusion, class_report


def predict_labels(eval_tasks, models):
    device = get_device(0)
    text = []
    true_labels_all = []
    pred_labels_all = []
    with torch.no_grad():
        for eval_task in eval_tasks:
            print(f'evaluating task {eval_task.task_name}... ')
            for mod in models:
                for fold in eval_task.get_folds()[0:1]: # predict labels of first fold only
                    for batch in fold.test:
                        tensor_dict_to_gpu(batch, device)
                        #
                        if len(mod.crf.per_task_output.values()) == 1:
                            #single task model
                            orig_task = batch["task"]
                            batch["task"] = list(mod.crf.per_task_output.keys())[0]
                            tensor_dict_to_gpu(batch, device)
                            output = mod(batch=batch, output_all_tasks=True)
                            batch["task"] = orig_task
                        else:
                            # multi-task model
                            tensor_dict_to_gpu(batch, device)
                            mod.to(device)
                            output = mod(batch=batch, output_all_tasks=True)
                        #
                        true_labels = batch["label_ids"].view(-1)
                        for task_output in output["task_outputs"]:
                            t = get_task(task_output["task"])
                            pred_labels = task_output["predicted_label"].view(-1)
                            cleared_true, cleared_predicted = clear_and_map_predicted_values(true_labels, pred_labels, eval_task.get_labels_titled(), t.get_labels_titled())
                            true_labels_all.append(cleared_true)
                            pred_labels_all.append(cleared_predicted)
                            text.append(batch['input_ids'])

                        tensor_dict_to_cpu(batch)
                        break
                    break
                break
            break
    return text, true_labels_all, pred_labels_all


def clear_and_map_padded_values(true_labels, predicted_labels, labels):
    assert len(true_labels) == len(predicted_labels)
    cleared_predicted = []
    cleared_true = []
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        # filter masked labels (0)
        if true_label > 0:
            cleared_true.append(labels[true_label])
            cleared_predicted.append(labels[predicted_label])
    return cleared_true, cleared_predicted


def clear_and_map_predicted_values(true_labels, predicted_labels, true_label_names, pred_label_names):
    assert len(true_labels) == len(predicted_labels)
    cleared_predicted = []
    cleared_true = []
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        # filter masked labels (0)
        if true_label > 0:
            cleared_true.append(true_label_names[true_label])
            cleared_predicted.append(pred_label_names[predicted_label])
    return cleared_true, cleared_predicted
