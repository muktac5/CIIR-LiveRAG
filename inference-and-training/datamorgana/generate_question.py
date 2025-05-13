import json
import time
import os
from typing import Dict, List
import requests
import random
import copy
import argparse

BASE_URL = "https://api.ai71.ai/v1/"
API_KEY = os.environ.get("API_KEY")

def check_budget():
    resp = requests.get(
        f"{BASE_URL}check_budget",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=4))

def wait_for_generation_to_finish(request_id: str):
    first_print = True
    while True:
        resp = requests.get(
            f"{BASE_URL}fetch_generation_results",
            headers={"Authorization": f"Bearer {API_KEY}"},
            params={"request_id": request_id},
        )
        resp.raise_for_status()
        if resp.json()["status"] == "completed":
            print('completed')
            print(json.dumps(resp.json(), indent=4))
            return resp.json()
        else:
            if first_print:
                first_print = False
                print("Waiting for generation to finish...", end='')
            else:
                print('.', end='')
            time.sleep(5)

def bulk_generate(n_questions: int, question_categorizations: List[Dict], user_categorizations: List[Dict]):
    resp = requests.post(
        f"{BASE_URL}bulk_generation",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
                "n_questions": n_questions,
                "question_categorizations": question_categorizations,
                "user_categorizations": user_categorizations
            }
    )
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    print(json.dumps(resp.json(), indent=4))

    result = wait_for_generation_to_finish(request_id)
    return result

def get_all_requests():
    resp = requests.get(
        f"{BASE_URL}get_all_requests",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    resp.raise_for_status()
    return resp.json()

def print_request_summary(requests):
    if 'data' not in requests:
        print('There are no requests')
    for request in requests['data']:
        print(f"{request['request_id']} : {request['status']}")

user_expertise_categorization = {
    "categorization_name": "user-expertise",
    "categories": [
        {
            "name": "expert",
            "description": "an expert on the subject discussed in the documents, therefore he asks complex questions.",
            "probability": 0.50
        },
        {
            "name": "novice",
            "description": "a person with very basic knowledge on the topic discussed in the topic. Therefore, he asks very simple questions.",
            "probability": 0.50
        }
    ]
}

question_phrasing_categorization = {
    "categorization_name": "question-phrasing-type",
    "categories": [
        {
            "name": "concise and natural",
            "description": "You should generate a concise, direct, and naturally phrased question consisting of a few words.",
            "probability": 0.35
        },
        {
            "name": "verbose and natural",
            "description": "You should generate a relatively long question consisting of more than 9 words, written in fluent, natural-sounding language.",
            "probability": 0.35
        },
        {
            "name": "short search query",
            "description": "You should generate a short query consists of less than 7 words phrased as a typed web query for search engines only keywords, without punctuation and without a natural-sounding structure.",
            "probability": 0.1
        },
        {
            "name": "long search query",
            "description": "You should generate a long query consists of more than 7 words phrased as a typed web query for search engines only keywords, without punctuation and without a natural-sounding structure.",
            "probability": 0.1
        },
        {
            "name": "concise and fancy",
            "description": "You should generate a short, well-structured question expressed in a stylistically rich or sophisticated manner.",
            "probability": 0.05
        },
        {
            "name": "verbose and fancy",
            "description": "You should generate a long and elaborate question phrased with refined, elevated, or formal language, often using complex sentence structures.",
            "probability": 0.05
        }
    ]
}

answer_phrasing_categorization = {
    "categorization_name": "answer-phrasing-type",
    "categories": [
    {
      "name": "concise and natural",
      "description": "Expects a brief, direct answer, typically a short phrase or sentence using everyday, accessible language.",
      "probability":0.4
    },
    {
      "name": "verbose and natural",
      "description": "Expects a detailed yet conversational response, typically one or more full sentences.",
      "probability":0.4
    },
    {
      "name": "concise and fancy",
      "description": "Expects a brief answer, typically one sentence or phrase, but expressed in an elevated, stylistically rich, or formal tone.",
      "probability":0.1
    },
    {
      "name": "verbose and fancy",
      "description": "Expects a longer and more elaborate answer, delivered in a refined, ornate, or sophisticated style, often using complex sentence structures and rich vocabulary.",
      "probability":0.1
    }
  ]
}

domain_categorization = { #notconsidered
    "categorization_name": "domain",
    "categories": [
        {
            "name": "artistic",
            "description":("An artistic question reflects creative taste, imagination, or aesthetic sensitivity."
                            "It may be inspired by artists, artworks, artistic movements, or forms of expression such as painting, music, literature, or design."
                            "These questions often use poetic, metaphorical, or emotionally rich language."
                            "Ex: What was Van Gogh trying to feel when he painted the sky in Starry Night?"),
            "probability":0.15
        },
        {
            "name": "philosophical",
            "description": ("A question that explores abstract, existential, or conceptual aspects of the content, such as the meaning, value, or implications of ideas."
            "These questions should only be generated if the document discusses reflective, moral, or human-centered themes."
            "If the document is purely factual or technical with no abstract dimension, skip this category."
            "Ex: What makes philosophy valuable even though it doesn't give us definite answers?"),
            "probability": 0.15
        },
        {
           "name":"technical",
           "description":("A question phrased in dense technical jargon, often highly specific or methodologically detailed."
                          "Example: What’s the difference between stochastic and deterministic dropout in Bayesian neural networks?"),
           "probability":0.15
        },
        {
            "name": "scientific",
            "description": "A question seeking a scientific explanation of natural or physical phenomena, often intended for educational purposes and expressed in accessible language. Example: Why do volcanoes erupt?",
            "probability": 0.15
        },
        {
            "name": "historical-cultural",
            "description": "A question concerning historical events, cultural contexts, or societal norms and values. Example: What impact did the Silk Road have on trade in Asia?",
            "probability": 0.15
        },
        {
            "name": "everyday-practical",
            "description": "A question about routine life, common tasks, or practical problem-solving. These are often found in how-to articles, guides, or advice columns. Example: How do I change a flat bicycle tire?",
            "probability": 0.15
        },
        {
            "name": "personal-reflective",
            "description": "A question focused on personal growth, emotional insight, or introspective curiosity. These often appear in personal blogs or wellness content. Example: Why do I feel anxious in social settings?",
            "probability": 0.10
        }
    ]
}

factuality_categorisation = { #multi-doc
    "categorization_name": "factuality",
    "categories": [
        {
            "name": "factoid",
            "description": "A question seeking a specific, concise piece of information or a short fact about a particular subject, such as a name, date, or number.",
            "probability": 0.50
        },
        {
            "name": "open-ended",
            "description": ("question inviting detailed or exploratory responses, encouraging discussion or elaboration." 
            "e.g., ‘what caused the French revolution?"),
            "probability": 0.50
        }
    ]
}

premise_categorization = { #multi-doc
    "categorization_name": "premise-categorization",
    "categories": [
      {
        "name": "without premise",
        "description": "a question that does not contain any premise or any information about the user.",
        "probability": 0.70
      },
      {
        "name": "with premise",
        "description": "a question starting with a very short premise, where the users reveal their needs or some information about themselves.",
        "probability": 0.30
      }
    ]
}

document_type_categorization = { #multi-doc
    "categorization_name": "linguistic-variation-type",
    "categories": [
        {
            "name": "similar-to-document",
            "description": "phrased using the same terminology and phrases appearing in the document (e.g., for the document 'The Amazon River has an average discharge of about 215,000–230,000 m3/s', 'what is the average discharge of the Amazon river')",
            "probability": 0.3
        },
        {
            "name": "distant-from-document",
            "description": "phrased using terms completely different from the ones appearing in the document (e.g., for a document 'The Amazon River has an average discharge of about 215,000–230,000 m3/s', 'How much water run through the Amazon?')",
            "probability": 0.4
        },
        {
            "name": "unpopular-entities",
            "description": "a question focusing on rare, less-documented, or emerging entities not widely mentioned in the fineweb corpora.",
            "probability": 0.3
        }
  ]       
}



question_intent_categorization = { #multi-doc
    "categorization_name": "question-intent-type",
    "categories": [
        {
            "name": "clarification",
            "description": "A question seeking further explanation or details about a specific concept, term, or methodology",
            "probability": 0.20,
        },
        {
            "name": "opinion",
            "description": "A question asking for a subjective viewpoint.",
            "probability": 0.20,
        },
        {
            "name": "comparison",
            "description": "A question that compares the information in the document to other studies, perspectives, or contexts.",
            "probability": 0.20,
        },
        {
            "name": "yes_no_question",
            "description": "A question expecting a yes or no answer",
            "probability": 0.20,
        },
        {
            "name":"hypothetical",
            "description": "A question imagining a what-if scenario, asking about potential futures, counterfactual histories, or theoretical cases. Use this category only when the document includes reflective or interpretive content such as commentary on history, society, science, technology, philosophy, or human behavior. Ex: What if climate data from the 1800s had been digitized?",
            "probability":0.20,
        }
    ]
}

question_aspect_categorisation = {  # multi-doc
    "categorization_name": "aspect-type",
    "categories": [
        {
            "name": "single-aspect",
            "description": "A question focused on one specific aspect or dimension of a concept or entity (e.g., What are the benefits of using AI in diagnostics?).",
            "probability": 0.50
        },
        {
            "name": "multi-aspect",
            "description": "A question about two different aspects of the same entity/concept (e.g., What are the advantages of AI-powered diagnostics, and what are the associated risks of bias in medical decision-making?).",
            "probability": 0.50
        }
    ]
}

question_turn_categorisation = { #multi-doc
  "categorization_name": "question-turn",
  "categories": [
    {
      "name": "single-turn",
      "description": "The initial question of a conversation, self-contained and understandable without any prior context.",
      "probability": 0.50,
    },
    {
      "name": "two-turn",
      "description": "A follow-up or compound question that either builds on previous context or combines two sub-questions on related or evolving concepts.",
      "probability": 0.50,
    }
  ]
}

multi_doc_categorization = {
    "categorization_name": "number-of-documents",
    "categories": [
        {
            "name": "multi-doc",
            "description": (
    " The information required to answer the question needs to come from two documents, specifically, "
    "the first document must provide information about the first entity/concept, while the second must "
    "provide information about the second entity/concept."),
            "probability": 0.5,
            "is_multi_doc": True
        },
        {
            "name": "single-doc",
            "description": (
    " The information required to answer the question can be found in a single document, "
    "which contains all the necessary information about the entity/concept."),
            "probability": 0.5,
            "is_multi_doc": False
        }
    ]
}

category_combinations = [
    [question_phrasing_categorization, answer_phrasing_categorization, multi_doc_categorization, factuality_categorisation, document_type_categorization, question_turn_categorisation, question_aspect_categorisation],
    [question_phrasing_categorization, answer_phrasing_categorization, multi_doc_categorization, premise_categorization, question_turn_categorisation, question_aspect_categorisation],
    [question_phrasing_categorization, answer_phrasing_categorization, multi_doc_categorization, question_intent_categorization, question_turn_categorisation, question_aspect_categorisation],
    [question_phrasing_categorization, answer_phrasing_categorization, multi_doc_categorization, premise_categorization, question_intent_categorization, question_turn_categorisation, question_aspect_categorisation],
    [question_phrasing_categorization, answer_phrasing_categorization, multi_doc_categorization, factuality_categorisation, premise_categorization, question_turn_categorisation, question_aspect_categorisation]
]

MULTI_DOC_EXPLANATION = (
    " The information required to answer the question needs to come from two documents, specifically, "
    "the first document must provide information about the first entity/concept, while the second must "
    "provide information about the second entity/concept."
)

multi_doc_applicable_cats = {
    "factuality",
    "linguistic-variation-type",
    "question-turn",
    "aspect-type"
}

def modify_categorizations_for_multi_doc(categorization_list):
    modified = []

    for cat in categorization_list:
        cat_copy = copy.deepcopy(cat)
        cat_name = cat_copy.get("categorization_name", "")

        for category in cat_copy.get("categories", []):
            if cat_name in multi_doc_applicable_cats:
                is_multi_doc = random.choice([True, False])
                category["is_multi_doc"] = is_multi_doc

                if is_multi_doc and MULTI_DOC_EXPLANATION not in category["description"]:
                    category["description"] += MULTI_DOC_EXPLANATION
            else:
                category["is_multi_doc"] = False 

        modified.append(cat_copy)

    return modified

parser = argparse.ArgumentParser(description="Generate questions using the AI71 API.")
parser.add_argument("--output_address", type=str, required=True, help="Output address for the generated data.")

if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = args.output_address
    os.makedirs(data_dir, exist_ok=True)

    for i, q_comb in enumerate(category_combinations, start=1):
        modified_comb = q_comb
        so_far_generated = 0
        target = 1600
        n_questions_per_batch = 500
        while so_far_generated < target:
            num_generation = min(n_questions_per_batch, target - so_far_generated)

            results = bulk_generate(
                n_questions=num_generation,
                question_categorizations=modified_comb,
                user_categorizations=[user_expertise_categorization]
            )

            response = requests.get(results["file"])
            qa_pairs = [json.loads(line) for line in response.text.splitlines()]

            batch_num = (so_far_generated // n_questions_per_batch) + 1
            filename = os.path.join(data_dir, f"qa_combo_{i}_batch_{batch_num}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

            so_far_generated += num_generation
            print(f"Saved {len(qa_pairs)} QA pairs to {filename}")