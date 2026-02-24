from datasets import Dataset, Features, Value, Image
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from evaluate import load
import datasets
import torch
import time
import numpy as np

import random
import numpy as np
import torch
import transformers

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)

"""

CREATE DATASETS

"""

def create_classification_dataset(DF_DATASET, MODELNAME):
    def hf_transform(example_batch):
        inputs = processor(
            [x.convert("RGB") for x in example_batch['img']], #
            return_tensors='pt' 
        )
        inputs['image_path'] = example_batch['image_path']
        inputs['labels'] = example_batch['labels']
        return inputs

    set_seed(1)
    img_path_list = DF_DATASET.image_path.to_list()
    label_list = DF_DATASET.label.to_list()

    assert len(img_path_list) == len(label_list)
    
    _CLASS_NAMES= list(np.unique(label_list))
    features=datasets.Features({
                      "image_path": datasets.Value("string"),
                      "img": datasets.Image(),
                      "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
                  })
    
    ds = datasets.Dataset.from_dict({"img":img_path_list, "image_path":img_path_list ,"label":label_list},features=features)
    ds = ds.rename_column("label", "labels")

    processor = AutoImageProcessor.from_pretrained(MODELNAME)
    prepared_ds = ds.with_transform(hf_transform)

    return prepared_ds


"""

TRAIN MODEL

"""

def train_hf_classification_model(outdir, epochs, batch_size, learning_rate, train_dataset, test_dataset, MODEL_NAME):
        def collate_fn(batch):
            return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
            }
        def custom_metrics(eval_pred):
            metric1 = load("precision")
            metric2 = load("recall")
            metric3 = load("f1")
            metric4 = load("accuracy")
            
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
        
            precision = metric1.compute(predictions=predictions, references=labels, average="micro")["precision"]
            recall = metric2.compute(predictions=predictions, references=labels, average="micro")["recall"]
            f1 = metric3.compute(predictions=predictions, references=labels, average="micro")["f1"]
            accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]
        
            return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

        set_seed(1)
        labels = train_dataset.features['labels'].names

        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, num_labels=len(labels), ignore_mismatched_sizes=True, id2label={str(i): c for i, c in enumerate(labels)}, label2id={c: str(i) for i, c in enumerate(labels)})
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Model on', device)
        model.to(device)

        early_stopping_patience_epochs = 3
        early_stopping = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_epochs)
        
        start_time = time.time()
        training_args = TrainingArguments(
        output_dir = outdir,
        disable_tqdm=True,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,   
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        no_cuda=False,
        fp16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=custom_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=processor,
            callbacks=[early_stopping]
        )

        train_results = trainer.train()
        num_epochs_completed = int(trainer.state.epoch)
        optimal_epochs_trained = num_epochs_completed if num_epochs_completed == epochs else num_epochs_completed - early_stopping_patience_epochs
        metrics = trainer.evaluate(test_dataset)
        return trainer, model, metrics, optimal_epochs_trained

