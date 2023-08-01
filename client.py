import requests
import json

template = """You are a teacher preparing questions for a quiz. Given the following document, please generate 1 multiple-choice questions (MCQs) with 4 options and a corresponding answer letter based on the document.
    Example question:
    Question: question here
    CHOICE_A: choice here
    CHOICE_B: choice here
    CHOICE_C: choice here
    CHOICE_D: choice here
    Answer: A or B or C or D
    <Begin Document>
    {doc}
    <End Document></s>"""


url = 'http://192.168.1.138:8000/predict'
headers = {'Content-Type': 'application/json'}

def predict(prompt: str) -> str:
    try:
        cleaned_prompt = " ".join(template.format(doc=prompt).split())
        data = {'prompt': cleaned_prompt}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response_data = response.json()

        if response.status_code == 200:
            return response_data["prediction"]
        else:
            # Instead of printing here, you might raise an exception or return a special value.
            return "Error: Status Code " + str(response.status_code)
    except requests.exceptions.RequestException as e:
        # Instead of printing here, you might raise an exception or return a special value.
        return "Error: " + str(e)

print(predict("In the summer of 1904, American chestnut trees in the Bronx were in trouble. Leaves, normally slender and brilliantly green, were curling at the edges and turning yellow. Some tree limbs and trunks sported rust-colored splotches."))

