# DocVQA Class

## Overview

The `DocVQA` class is designed to handle document-based question-answering tasks. It leverages PaddleNLP's Taskflow from the `paddlenlp` library to answer questions based on provided documents, including images and PDF files.

## Class Methods

#### `__init__(self)`

Initializes the `DocVQA` class and sets up the PaddleNLP Taskflow for document intelligence.

#### `answer_questions(self, file_type: str, file_path: str, questions: List[str]) -> List[Dict[str, Any]]`

Answers a list of questions based on the provided document.

#### Arguments:
- `file_type`: Type of the document (`image` or `pdf`).
- `file_path`: File path of the document (`image` or `pdf`).
- `questions`: List of questions to be answered.

#### Returns:
- List of dictionaries where each element contains the original prompt and the answer.

#### Example:
``` python

from doc_intelligence.model.qustion_answering.document_question_answering import DocVQA

# get instance
doc_vqa = DocVQA()
answers = doc_vqa.answer_questions(file_type="image", file_path="./docs/images/document_qa_example.webp", questions=["what is the name of the patient?", "What is the date of admission?", "What is the name of the doctor?", "What was the diagnosis?"]) 
```

<figure markdown>
  ![Example of Discharge Summary for DocVQA](./images/document_qa_example.webp){ width="700" }
  <figcaption>Example of image (Discharge Summary) for DocVQA</figcaption>
</figure>

```json
{ "prompt": "What is the name of the patient?", "result": [{ "value": "Mr.SONAIMUTHU", "prob": 1, "start": 15, "end": 17 }]}, 
{ "prompt": "What is the date of admission?", "result": [ { "value": "21/01/2019", "prob": 1, "start": 37, "end": 41 }]},
{ "prompt": "What is the name of the doctor?", "result": [ { "value": "DR.AnandMBBS,MD.", "prob": 0.92, "start": 54, "end": 59}]},
{ "prompt": "What was the diagnosis?", "result": [ { "value": "LOWER OESOPHAGEAL HIATUS HERNIA", "prob": 1, "start": 62, "end": 68 }]}
```


