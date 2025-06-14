import os
import pandas as pd
import re
from pathlib import Path
from datasets import Dataset, DatasetDict, ClassLabel, Sequence, Value, Features
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent        # .../project/src/dataloader
PROJECT_ROOT = HERE.parent.parent             # .../project

def get_data(name):
    if name == "Phenotype":
        return Phenotype()
    elif name == "ChemProt":
        return ChemProt()
    elif name == "DEID":
        return DEID()
    elif name == "COS":
        return COS()
    elif name == "SocialHistory":
        return SocialHistory()
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")

class Phenotype:
    def __init__(self):
        # The phenotype columns (exclude the "NONE" column as it only indicates "no phenotype")
        self.class_names = [
            'ADVANCED.CANCER', 'ADVANCED.HEART.DISEASE', 'ADVANCED.LUNG.DISEASE',
            'ALCOHOL.ABUSE', 'CHRONIC.NEUROLOGICAL.DYSTROPHIES', 'CHRONIC.PAIN.FIBROMYALGIA',
            'DEMENTIA', 'DEPRESSION', 'DEVELOPMENTAL.DELAY.RETARDATION', 'NON.ADHERENCE',
            'OBESITY', 'OTHER.SUBSTANCE.ABUSE',
            'SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS', 'UNSURE'
        ]
        self.is_entailment = False
        self.num_labels = len(self.class_names)
        self.problem_type = "multi_label_classification"
        self.cache_dir = os.path.join(f"{PROJECT_ROOT}/data/processed/phenotype")
        self.dataset = self.preprocess_data()

    def preprocess_data(self):
        # if we've cached already, just load
        if os.path.isdir(self.cache_dir):
            return DatasetDict.load_from_disk(self.cache_dir)
        # 1) Load the annotation and notes CSVs
        df_ann = pd.read_csv(f"{PROJECT_ROOT}/data/raw/phenotype/ACTdb102003.csv")
        df_mimic = pd.read_csv(f"{PROJECT_ROOT}/data/raw/phenotype/NOTEEVENTS.csv")

        # 2) Filter out inconsistent rows (NONE=1 & any phenotype=1)
        mask = ~((df_ann['NONE'] == 1) & (df_ann[self.class_names].sum(axis=1) > 0))
        df_ann = df_ann[mask].copy()

        # 3) Build the multi-hot "labels" vector of length len(class_names)
        df_ann['labels'] = df_ann.apply(self.make_multi_hot, axis=1)

        # 4) Merge with the text from NOTEEVENTS
        df = df_ann.merge(
            df_mimic[['ROW_ID', 'TEXT']],
            on='ROW_ID', how='inner'
        )[['TEXT', 'labels']]

        # 5) Rename for HF Dataset
        df = df.rename(columns={'TEXT': 'text'})

        # 6) Convert to HF Dataset and split
        ds = Dataset.from_pandas(df.reset_index(drop=True))

        # first split off 20% for test
        split1 = ds.train_test_split(test_size=0.2, seed=42)
        train_val = split1['train']
        test_ds   = split1['test']
        # then split train_val (10% of total)
        split2 = train_val.train_test_split(test_size=0.125, seed=42)
        train_ds = split2['train']
        val_ds   = split2['test']

        dataset_dict = DatasetDict({
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        })

        # 6) cache to disk
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset_dict.save_to_disk(self.cache_dir)

        return dataset_dict
    
    def make_multi_hot(self, row):
        # if NONE=1, then no phenotype => all zeros
        if row['NONE'] == 1:
            return [0.0] * self.num_labels
        else:
            # cast each entry to float
            return [float(x) for x in row[self.class_names].tolist()]


class ChemProt:
    def __init__(self):
        self.is_entailment = False
        self.problem_type = "single_label_classification"
        # where to cache the processed HuggingFace dataset
        self.cache_dir = os.path.join(f"{PROJECT_ROOT}/data/processed/ChemProt")
        # load (or preprocess & cache)
        self.dataset = self.preprocess_data()
        # after dataset is ready, set class names and num_labels
        self.class_names = self.dataset["train"].features["labels"].names
        self.num_labels = len(self.class_names)

    def preprocess_data(self):
        # 1) load from cache if available
        if os.path.isdir(self.cache_dir):
            return DatasetDict.load_from_disk(self.cache_dir)

        # 2) read train/dev/test TSVs
        splits = {}
        for split in ["train", "dev", "test"]:
            path = os.path.join(f"{PROJECT_ROOT}/data/raw/ChemProt/{split}.tsv")
            df = pd.read_csv(path, sep="\t")
            # rename columns
            df = df.rename(columns={"sentence": "text", "label": "label_str"})
            splits[split] = df

        # 4) convert each DataFrame to a Dataset, mapping labels
        ds_splits = {}
        for split, df in splits.items():
            ds = Dataset.from_pandas(df, preserve_index=False)
            # cast the string column to ClassLabel
            ds = ds.class_encode_column(
                column="label_str"
            )
            # now the label column is named "label_str" → rename to "labels"
            ds = ds.rename_column("label_str", "labels")
            ds_splits[split] = ds

        # 5) assemble DatasetDict, mapping "dev"→"validation"
        dataset_dict = DatasetDict({
            "train": ds_splits["train"],
            "validation": ds_splits["dev"],
            "test": ds_splits["test"]
        })

        # 6) cache to disk
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset_dict.save_to_disk(self.cache_dir)

        return dataset_dict


class DEID:
    def __init__(self):
        self.cache_dir = os.path.join(f"{PROJECT_ROOT}/data/processed/DEID")
        self.problem_type = "token_classification"
        self.id2label = {0: 'O',
                         1: 'B-age',
                         2: 'B-date',
                         3: 'B-location',
                         4: 'B-name',
                         5: 'B-other',
                         6: 'B-phone_nb',
                         7: 'I-age',
                         8: 'I-date',
                         9: 'I-location',
                         10: 'I-name',
                         11: 'I-other',
                         12: 'I-phone_nb'
                         }
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.num_labels = len(self.id2label)
        self.dataset = self.preprocess_data()

    def preprocess_data(self):
        # 1. If we've cached already, just load
        if os.path.isdir(self.cache_dir):
            return DatasetDict.load_from_disk(self.cache_dir)

        # 2. Load raw records
        records_txt = self.load_records(f"{PROJECT_ROOT}/data/raw/DEID/id.text")
        records_res = self.load_records(f"{PROJECT_ROOT}/data/raw/DEID/id.res")
        if len(records_txt) != len(records_res):
            print("Warning: The number of records in id.txt and id.res do not match.")

        # 3. Turn each pair of records into a token‑level example
        examples = []
        for rec_txt, rec_res in zip(records_txt, records_res):
            token_tag_pairs = self.process_record(rec_txt, rec_res)
            tokens, tags = zip(*token_tag_pairs)
            examples.append({
                "tokens": list(tokens),
                "ner_tags": tags
            })

        # 4. Split into train / validation / test
        train_ex, temp_ex = train_test_split(examples, test_size=0.30, random_state=42)
        dev_ex,  test_ex = train_test_split(temp_ex, test_size=0.50, random_state=42)

        # 5. Build a DatasetDict
        ds = DatasetDict({
            "train":      Dataset.from_list(train_ex),
            "validation": Dataset.from_list(dev_ex),
            "test":       Dataset.from_list(test_ex),
        })

        label_list = [ self.id2label[i] for i in range(self.num_labels) ]
        class_label = ClassLabel(names=label_list)

        features = Features({
            "tokens": Sequence(feature=Value("string")),
            "ner_tags": Sequence(feature=class_label),
        })

        ds = ds.cast(features)

        # 6. Cache to disk and return
        os.makedirs(self.cache_dir, exist_ok=True)
        ds.save_to_disk(self.cache_dir)
        return ds


    def write_records(self, records, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            # Records in CoNLL format are separated by a blank line.
            for record in records:
                f.write(record + "\n\n")
        
    def load_records(self, filename):
        """
        Read the entire file and extract records.
        
        Each record is assumed to start with a line beginning with
        "START_OF_RECORD=" and end with "||||END_OF_RECORD". The regex
        uses a non-greedy match (.*?) to capture all content between these markers.
        The DOTALL flag ensures that newline characters are included in the match.
        """
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex pattern to match each record including the markers.
        pattern = r'(START_OF_RECORD=.*?\|\|\|\|END_OF_RECORD)'
        records = re.findall(pattern, content, flags=re.DOTALL)
        return records

    def extract_masks(self, record):
        """
        Given a record string, extract all masks.
        
        Masks in the file are in the form: [**Some Text Here**]
        The regex uses non-greedy matching (.*?) to extract the mask contents.
        """
        mask_pattern = r'\[\*\*.*?\*\*\]'
        return re.findall(mask_pattern, record)
    
    def process_record(self, text_txt, text_res):
        """
        Given a record from the original file (id.txt) and the corresponding record from
        the masked file (id.res), produce a list of (token, tag) tuples in CoNLL format.
        
        Steps:
        1. Remove record markers and any header number pattern (e.g. "1||||1||||") from both texts.
        2. Replace newlines with a space.
        3. Split the masked text (text_res) into alternating plain text and mask segments.
        4. Traverse text_txt using a pointer. Plain text segments are tokenized and tagged as 'O'.
            For each mask segment, the corresponding span in text_txt is tokenized;
            its first token is tagged B-<category> and the remainder I-<category>.
        """
        # 1. Remove record markers.
        for marker in ["START_OF_RECORD=", "||||END_OF_RECORD"]:
            text_txt = text_txt.replace(marker, "")
            text_res = text_res.replace(marker, "")
        
        # Remove header pattern left from beginning, e.g., "1||||1||||"
        text_txt = re.sub(r'^\d+\|\|\|\|\d+\|\|\|\|', '', text_txt)
        text_res = re.sub(r'^\d+\|\|\|\|\d+\|\|\|\|', '', text_res)
        
        text_txt = text_txt.strip()
        text_res = text_res.strip()
        
        # 2. Replace newlines with a space.
        text_txt = text_txt.replace("\n", " ")
        text_res = text_res.replace("\n", " ")
        
        # 3. Split the masked record into segments; even indices: plain text, odd: mask placeholders.
        segments = re.split(r'(\[\*\*.*?\*\*\])', text_res)
        
        pointer = 0  # pointer in text_txt
        tokens_with_tags = []
        
        for i, segment in enumerate(segments):
            if i % 2 == 0:
                # Plain text segment.
                plain = segment
                for token in plain.split():
                    tokens_with_tags.append((token, "O"))
                pointer += len(plain)
            else:
                # Mask placeholder segment.
                cat = self.map_mask(segment)
                # Look ahead to the next plain segment to determine the extent of the masked span.
                next_plain = segments[i+1] if (i+1) < len(segments) else ""
                if next_plain:
                    pos = text_txt.find(next_plain, pointer)
                    if pos == -1:
                        pos = len(text_txt)
                else:
                    pos = len(text_txt)
                masked_text = text_txt[pointer:pos]
                masked_tokens = masked_text.split()
                if masked_tokens:
                    b_tag = f"B-{cat}" if cat is not None else "B"
                    tokens_with_tags.append((masked_tokens[0], b_tag))
                    for token in masked_tokens[1:]:
                        i_tag = f"I-{cat}" if cat is not None else "I"
                        tokens_with_tags.append((token, i_tag))
                pointer = pos
        return tokens_with_tags
    
    def map_mask(self, mask):
        """
        Map a given mask to a category based on the mapping rules.
        
        Updated rules:
        - If the mask (case-insensitive) contains "name", return "name".
        - If the mask contains "e-mail address" (or "email address"), return "other".
        - If the mask contains "telephone/fax" or "pager number", return "phone_nb".
        - If the mask contains "age over", return "age".
        - If the mask contains "location", return "location".
        - Also, if it contains any of the location-specific keywords ("hospital", "street address", "unit number"), map to "location".
        - If the mask is numeric, looks like a date (e.g. "01-01", "1959", "1901-12-16"), or contains "date range" or "month/year", return "date".
        - If the mask is blank or contains "company", return "other".
        - Otherwise, return None.
        """
        # Remove the markers and trim whitespace.
        content = mask[3:-3].strip()
        lower_content = content.lower()
        
        # Priority for name: if the text contains "name", map to "name".
        if "name" in lower_content:
            return "name"
        
        # Explicitly check for e-mail address and map to other.
        if "e-mail address" in lower_content or "email address" in lower_content:
            return "other"
        
        # Rule for phone number: if the text contains "telephone/fax" or "pager number".
        if "telephone/fax" in lower_content or "pager number" in lower_content:
            return "phone_nb"
        
        # Rule for age.
        if "age over" in lower_content:
            return "age"
        
        # Rule for location: if the text contains "location".
        if "location" in lower_content:
            return "location"
        
        # Additional location-specific keywords.
        for keyword in ["hospital", "street address", "unit number"]:
            if keyword in lower_content:
                return "location"
        
        # Rule for date: check for purely numeric values or typical date patterns.
        if (re.fullmatch(r'\d+', content) or 
            re.fullmatch(r'\d{1,2}-\d{1,2}', content) or 
            re.fullmatch(r'\d{4}', content) or 
            re.fullmatch(r'\d{4}-\d{2}-\d{2}', content) or 
            "date range" in lower_content or 
            "month/year" in lower_content):
            return "date"
        
        # Rule for other: if the mask is blank or contains "company".
        if content == "" or "company" in lower_content:
            return "other"
        
        # Return None if no category fits.
        return None
    

class COS:
    def __init__(self):
        # Problem setup
        self.problem_type = "token_classification"
        self.id2label = {0: 'B-Attr', 1: 'B-Cos', 2: 'B-Loc', 3: 'B-Ref',
                         4: 'B-Val', 5: 'I-Attr', 6: 'I-Cos', 7: 'I-Loc',
                         8: 'I-Ref', 9: 'I-Val', 10: 'O'}
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.num_labels = len(self.id2label)

        # Load the dataset
        self.dataset = self.load_brat_as_dataset(f"{PROJECT_ROOT}/data/raw/COS")

    def process_ann_only(self, ann_path):
        """
        Reads only the .ann file in two passes:
         1) Collect all Snippet entries (record boundaries + text)
         2) For each snippet, re-read the file to gather all Cos/Ref/Loc/Attr/Val spans
            that fall within that snippet, tokenize, and assign BIO labels.
        Returns a list of {"tokens": [...], "ner_tags": [...]} examples.
        """
        # --- PASS 1: collect snippet boundaries & text ---
        snippets = []
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("T"):
                    continue
                tid, rest = line.split("\t", 1)
                parts = rest.split()
                if parts[0] != "Snippet" or len(parts) < 4:
                    continue
                start, end = map(int, parts[1:3])
                text = rest.split(None, 3)[-1].strip()
                snippets.append({"start": start, "end": end, "text": text})

        examples = []
        # --- PASS 2: for each snippet, collect features & BIO-tag ---
        for snip in snippets:
            s_start, s_end, s_text = snip["start"], snip["end"], snip["text"]

            # gather relevant feature spans
            feats = []
            with open(ann_path, encoding="utf-8") as f2:
                for line in f2:
                    if not line.startswith("T"):
                        continue
                    tid, rest = line.split("\t", 1)
                    parts = rest.split()
                    label = parts[0]
                    if label not in {"Cos", "Ref", "Loc", "Attr", "Val"} or len(parts) < 4:
                        continue
                    start, end = map(int, parts[1:3])
                    # only keep those inside the snippet span
                    if start < s_start or end > s_end:
                        continue
                    feats.append({
                        "label": label,
                        "start": start - s_start,  # relative to snippet
                        "end":   end   - s_start
                    })

            # tokenize snippet text, splitting off punctuation
            tokens, spans = [], []
            for m in re.finditer(r"\w+|[^\w\s]", s_text):
                tokens.append(m.group())
                spans.append((m.start(), m.end()))

            # initialize all-O tags
            ner_tags = ["O"] * len(tokens)

            # assign BIO tags
            for feat in feats:
                idxs = [
                    i for i, (ts, te) in enumerate(spans)
                    if ts >= feat["start"] and te <= feat["end"]
                ]
                if not idxs:
                    continue
                ner_tags[idxs[0]] = f"B-{feat['label']}"
                for i in idxs[1:]:
                    ner_tags[i] = f"I-{feat['label']}"

            examples.append({
                "tokens":   tokens,
                "ner_tags": ner_tags
            })

        return examples

    def load_brat_as_dataset(self, folder):
        # collect all examples from every .ann file
        all_exs = []
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".ann"):
                continue
            ann_path = os.path.join(folder, fn)
            all_exs.extend(self.process_ann_only(ann_path))

        # build HuggingFace DatasetDict splits
        ds = Dataset.from_list(all_exs)
        split1 = ds.train_test_split(test_size=0.3, seed=42)
        train = split1["train"]
        rest  = split1["test"].train_test_split(test_size=0.5, seed=42)
        eval_ = rest["train"]
        test  = rest["test"]
        ds = DatasetDict({
            "train":      train,
            "validation": eval_,
            "test":       test
        })

        # cast ner_tags to ClassLabel
        label_list  = [ self.id2label[i] for i in range(self.num_labels) ]
        class_label = ClassLabel(names=label_list)
        features    = Features({
            "tokens":   Sequence(feature=Value("string")),
            "ner_tags": Sequence(feature=class_label)
        })
        ds = ds.cast(features)

        return ds


class SocialHistory:
    def __init__(self):
        self.cache_dir = os.path.join(f"{PROJECT_ROOT}/data/processed/SocialHistory")
        self.problem_type = "token_classification"
        self.id2label = {0: 'B-Alcohol', 1: 'B-Amount', 2: 'B-Drug', 3: 'B-EnvironmentalExposure', 4: 'B-ExposureHistory', 5: 'B-Extent',
                         6: 'B-Family', 7: 'B-Frequency', 8: 'B-History', 9: 'B-InfectiousDiseases', 10: 'B-LivingStatus', 11: 'B-Location',
                         12: 'B-MaritalStatus', 13: 'B-MedicalCondition', 14: 'B-Method', 15: 'B-Occupation', 16: 'B-Other', 17: 'B-PhysicalActivity',
                         18: 'B-QuitHistory', 19: 'B-Residence', 20: 'B-SexualHistory', 21: 'B-Status', 22: 'B-Temporal', 23: 'B-Tobacco',
                         24: 'B-Type', 25: 'I-Alcohol', 26: 'I-Amount', 27: 'I-Drug', 28: 'I-EnvironmentalExposure', 29: 'I-ExposureHistory',
                         30: 'I-Extent', 31: 'I-Family', 32: 'I-Frequency', 33: 'I-History', 34: 'I-InfectiousDiseases', 35: 'I-LivingSituation',
                         36: 'I-LivingStatus', 37: 'I-Location', 38: 'I-MaritalStatus', 39: 'I-MedicalCondition', 40: 'I-Method', 41: 'I-Occupation',
                         42: 'I-Other', 43: 'I-QuitHistory', 44: 'I-Residence', 45: 'I-SexualHistory', 46: 'I-Status', 47: 'I-Temporal', 48: 'I-Tobacco',
                         49: 'I-Type', 50: 'O'}
        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.num_labels = len(self.id2label)
        self.dataset = self.load_brat_as_dataset(f"{PROJECT_ROOT}/data/raw/SocialHistory")

    def parse_ann_file(self, ann_path):
        """Read only text-bound annotations (lines starting with 'T')."""
        anns = []
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("T"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                meta = parts[1].split()
                if len(meta) < 3:
                    continue
                typ = meta[0]
                try:
                    start, end = map(int, meta[1:3])
                except ValueError:
                    continue
                anns.append({"start": start, "end": end, "type": typ})
        return sorted(anns, key=lambda x: x["start"])

    def process_pair(self, txt_path, ann_path):
        """
        Returns a dict with:
        - "tokens":  list[str]
        - "ner_tags": list[str]  (BIO labels)
        """
        text = open(txt_path, encoding="utf-8").read()
        anns = self.parse_ann_file(ann_path)

        # split on whitespace, keep offsets
        tokens = [(m.group(), m.start(), m.end())
                for m in re.finditer(r"\S+", text)]
        labels = ["O"] * len(tokens)

        # assign BIO
        for ann in anns:
            st, en, typ = ann["start"], ann["end"], ann["type"]
            idxs = [i for i,(tok, s,e) in enumerate(tokens) if s>=st and e<=en]
            if not idxs:
                continue
            labels[idxs[0]] = f"B-{typ}"
            for i in idxs[1:]:
                labels[i] = f"I-{typ}"

        # strip offsets, return tokens + labels
        token_strs = [tok for tok,_,_ in tokens]
        return {"tokens": token_strs, "ner_tags": labels}

    def load_brat_as_dataset(self, folder):
        # collect all examples
        examples = []
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".txt"):
                continue
            base = fn[:-4]
            txt, ann = os.path.join(folder, fn), os.path.join(folder, base+".ann")
            if not os.path.exists(ann):
                print(f"Missing .ann for {fn}, skipping.")
                continue
            examples.append(self.process_pair(txt, ann))

        # turn into a Dataset and split
        ds = Dataset.from_list(examples)
        split1 = ds.train_test_split(test_size=0.3, seed=42)
        train = split1["train"]
        rest  = split1["test"].train_test_split(test_size=0.5, seed=42)
        eval_ = rest["train"]
        test  = rest["test"]
        ds = DatasetDict({
            "train": train,
            "validation": eval_,
            "test": test
        })
        label_list = [ self.id2label[i] for i in range(self.num_labels) ]
        class_label = ClassLabel(names=label_list)

        features = Features({
            "tokens": Sequence(feature=Value("string")),
            "ner_tags": Sequence(feature=class_label),
        })

        ds = ds.cast(features)
        return ds