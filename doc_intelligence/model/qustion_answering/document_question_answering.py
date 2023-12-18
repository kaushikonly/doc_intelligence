from typing import Any, Dict, List

import fitz
from paddlenlp import Taskflow
from common.utils import files


class DocVQA:

    def __init__(self):
        """
        from paddlenlp import Taskflow
        docprompt = Taskflow("document_intelligence")
        # Types of doc: A string containing a local path to an image
        docprompt({"doc": "./invoice.jpg", "prompt": ["what is total amount?", "what is invoice number?"]})
        """

        # Get PaddleNLP Taskflow pipeline
        self.__docprompt = Taskflow("document_intelligence", lang="en")

    def answer_questions(
        self, file_type: str, file_path: str, questions: List[str], 
    ):
        """
        from document_intelligence.model.question_answering import DocVQA
        docprompt = DocVQA(lang="en")
        docprompt.answer_questions(file_type="image", file_path="./invoice.jpg", questions=["what is total amount?", "what is invoice number?"])

        
        # Types of doc: A string containing a http link pointing to an image
        docprompt({"doc": "../../../data/document_question_answering/discharge-summary-template-18.jpg", "prompt": ["what is the name of the patient?", "what is the final dignosis?"]})
        '''
        [{'prompt': 'what is the name of the patient?', 'result': [{'value': 'PatientHSampleProvider', 'prob': 0.96, 'start': 11, 'end': 11}]}, {'prompt': 'what is the final dignosis?', 'result': [{'value': '', 'prob': 0.0, 'start': -1, 'end': -1}]}]
        '''
        
        Args: 
            file_type: image/pdf. type of the document  
            file_path: file path of the document image/pdf
            questions: lisf of questions to be answered
        Returns: 
            List of dicts where each elements contains original promot and the answer of the given prompt
            example - [{"prompt" : "your prompt", "result", [{"value" : "answer", "prob" : 0.9, "start": 23, "end": 42}]}]
        """
        if file_type == "image":
            query = {"doc": file_path, "prompt": questions}
            answers = self.__docprompt([query])

        if file_type == "pdf":
            answers = self.__extract_answers_from_pdf_pages(
                file_path, questions
            )

        return answers

    def __extract_answers_from_pdf_pages(
        self, file_path: str, questions: List[str]
    ) -> List[Dict]:

        """
        Get the highest probablity answer from the output of multiple pages of pdf

        Args:
            file_path: image or pdf file path
            questions: list of questions to be answered
        Returns:
            List of dicts where each elements contains original promot and the answer of the given prompt
            example: [{'prompt': 'what is the name of the patient?', 'result': [{'value': 'Mr Rakesh Sharma', 'prob': 0.74, 'start': 23, 'end': 34}]}]

        """
        extracted_answers = []

        doc = fitz.open(file_path)

        for _, page in enumerate(doc):
            # Save pdf page as image
            pix = page.get_pixmap()
            pdf_page_image_path = files.get_temp_file_path("../../../data/document_question_answering/pdf_page.png")
            pix.save(pdf_page_image_path)

            query = {
                "doc": pdf_page_image_path,
                "prompt": [que for que in questions],
            }

            # Extract answers from image
            answers = self.__docprompt([query])
            extracted_answers.append(answers)

        return self.__choose_most_probable_answer(extracted_answers)

    def __choose_most_probable_answer(
        self, answers_lists: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:

        final_responses = []

        for index in range(len(answers_lists[0])):
            max_prob = float("-inf")
            max_response = None

            for answers_list in answers_lists:
                response = answers_list[index]
                prob = response.get("result", [{}])[0].get("prob", 0)

                if prob > max_prob:
                    max_prob = prob
                    max_response = response

            final_responses.append(max_response)

        return final_responses

    def __post_process_results(
        self, answer : List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        """
        perform any post processing on the answers
        """
        # answer = answer.replace("name", "")
        # answer = answer.replace("admission", "")
        # answer = answer.replace(",", "")
        # answer = answer.title()
        return answer