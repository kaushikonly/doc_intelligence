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
answers = doc_vqa.answer_questions(file_type="image", file_path="./invoice.jpg", questions=["what is total amount?", "what is invoice number?"]) 
```
