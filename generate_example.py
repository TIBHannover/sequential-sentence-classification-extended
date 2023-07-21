import torch
from transformers import BertTokenizer

from utils import tensor_dict_to_gpu, tensor_dict_to_cpu, get_device
from models import BertHSLN, BertHSLNMultiSeparateLayers
from eval import *
from task import *



def create_task(create_func):
    return create_func(train_batch_size=32, max_docs=-1)


def get_all_tasks():
    return [
        create_task(pubmed_task),
        create_task(nicta_task),
        create_task(dri_task),
        create_task(art_task),
    ]


def predict_labels(eval_tasks, models):
    #labels_matrix = get_labels_matrix(get_all_tasks())
    device = get_device(0)
    examples = []

    BERT_VOCAB = "bert_model/scibert_scivocab_uncased/vocab.txt"
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    with torch.no_grad():
        for eval_task in eval_tasks:
            print(f'evaluating task {eval_task.task_name}... ')
            for mod in models:
                for fold in eval_task.get_folds()[0:1]: # predict labels of first fold only
                    for batch in fold.test[:4]:
                        tensor_dict_to_gpu(batch, device)
                        if type(mod) == BertHSLNMultiSeparateLayers:
                            mod.to_device(device, device)
                        else:
                            mod.to(device)
                        print('generate_example', batch['task'])
                        output = mod(batch=batch, output_all_tasks=True)

                        true_labels = batch["label_ids"].view(-1)
                        for task_output in output["task_outputs"]:
                            if task_output['task'] != eval_task.task_name:
                                continue
                            t = get_task(task_output["task"])
                            pred_labels = task_output["predicted_label"].view(-1)
                            print(t.get_labels_titled())
                            print(eval_task.get_labels_titled())
                            cleared_true, cleared_predicted = clear_and_map_predicted_values(true_labels, pred_labels, eval_task.get_labels_titled(), t.get_labels_titled())
                            examples.append({
                                'text': tokenizer.batch_decode(batch['input_ids'][0]),
                                'true_labels': cleared_true,
                                'pred_labels': cleared_predicted,
                                'task': task_output['task']
                            })

                        tensor_dict_to_cpu(batch)
    return examples


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


def load_model(path, tasks, model_class=BertHSLN, config=dict()):
    BERT_MODEL = "bert_model/scibert_scivocab_uncased/"
    best_config = {
        "bert_model": BERT_MODEL,
        "bert_trainable": False,
        "model": BertHSLN.__name__,
        "cacheable_tasks": [],

        "dropout": 0.5,
        "word_lstm_hs": 758,
        "att_pooling_dim_ctx": 200,
        "att_pooling_num_ctx": 15,

        "lr": 3e-05,
        "lr_epoch_decay": 0.9,
        "batch_size":  32,
        "max_seq_length": 128,
        "max_epochs": 20,
        "early_stopping": 5
    }
    best_config.update(config)
    
    model = model_class(best_config, tasks)
    params = torch.load(path, map_location=torch.device("cuda"))
    model.load_state_dict(params)        
    return model


def generate_example():
    multi_all_model = load_model("/nfs/home/stamatakism/sentence_project/results/2023-04-27_10_43_27_mult_all/0_0_19_model.pt", get_all_tasks(), BertHSLN)
    examples = predict_labels(get_all_tasks(), [multi_all_model])

    with open('output_mult_all.txt', 'w') as f:
        for example in examples:
            for i in range(len(example['text'])):
                text = example['text'][i].replace(' [PAD]', '').replace('[CLS] ', '').replace('[SEP]', '')
                f.write(f"{example['task']} {example['true_labels'][i]} {example['pred_labels'][i]} : {text}\n")
            f.write('\n\n\n')


def generate_example_grp_all():
    multi_groups_model_config = dict()
    multi_groups_model_config["attention_groups"] = [[PUBMED_TASK, NICTA_TASK, ART_TASK, DRI_TASK]]
    multi_groups_model_config["sentence_encoder_groups"] = [[PUBMED_TASK, NICTA_TASK], [ART_TASK, DRI_TASK]]
    multi_groups_model_config["output_groups"] = [[PUBMED_TASK], [NICTA_TASK], [ART_TASK], [DRI_TASK]]

    multi_groups_model_config["context_enriching_groups"] = [[PUBMED_TASK, NICTA_TASK], [ART_TASK, DRI_TASK]]

    multi_groups_model = load_model("/nfs/home/stamatakism/sentence_project/results/2023-04-20_09_24_28_mult_grouped/0_0_19_model.pt", get_all_tasks(), BertHSLNMultiSeparateLayers, multi_groups_model_config)

    examples = predict_labels(get_all_tasks(), [multi_groups_model])

    with open('output_mult_group.txt', 'w') as f:
        for example in examples:
            for i in range(len(example['text'])):
                text = example['text'][i].replace(' [PAD]', '').replace('[CLS] ', '').replace('[SEP]', '')
                f.write(f"{example['task']} {example['true_labels'][i]} {example['pred_labels'][i]} : {text}\n")
            f.write('\n\n\n')


def annotate(text):
    colors = ['hlcyan', 'hl', 'hlblue', 'hlpurple', 'hlgray', 'hlbrown']
    text = text.split('\n')
    text = [t.split(' ', 3) for t in text]
    classes = {t[1] for t in text}
    classes = classes.union({t[2] for t in text})
    class_map = {class_: color for (class_, color) in zip(classes, colors)}
    for sentence in text:
        print(f'\{class_map[sentence[1]]}{{{sentence[3][2:]}}}')
    print('\n\n')
    for sentence in text:
        print(f'\{class_map[sentence[2]]}{{{sentence[3][2:]}}}')

