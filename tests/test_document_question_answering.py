import PIL
import pytest
from doc_intelligence.model.qustion_answering.document_question_answering import DocVQA

# get instance
doc_vqa = DocVQA()

def test_doc_vqa_image(): 

    answers = doc_vqa.answer_questions(file_type="image", file_path="./data/document_question_answering/discharge-summary-template-18.jpg", questions=["what is name of the patient?", "what is age of the patient?"]) 
    assert isinstance(answers, list) and len(answers) > 0

def test_doc_vqa_pdf(): 

    answers = doc_vqa.answer_questions(file_type="image", file_path="./data/document_question_answering/discharge_summary_pdf_example.pdf", questions=["what is name of the patient?", "what is age of the patient?"]) 
    assert isinstance(answers, list) and len(answers) > 0


def test_invalid_image_file_type(): 

    with pytest.raises(ValueError):
        doc_vqa.answer_questions(file_type="audio", file_path="./data/document_question_answering/discharge-summary-template-18.jpg", questions=["what is name of the patient?", "what is age of the patient?"]) 
