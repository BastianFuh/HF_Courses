"""Based on the Huggingface Course for NLP Processing. See, https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt"""

from other.util import pipeline, pretty_print


#
#    Initializing a model and calling it
#
if False:
    # Initilize the model
    classifier = pipeline("sentiment-analysis")

    # Call the model with one input
    pretty_print(
        classifier("I've been waiting for a HuggingFace course my whole life.")
    )

    # Call the model with multiple arguments, will produce two outputs
    pretty_print(
        classifier(
            [
                "I've been waiting for a HuggingFace course my whole life.",
                "I hate this so much!",
            ]
        )
    )

#
#    Zero-shot classification.
#    In contrast to a model with fixed output labels, zero shot classification uses a set of provided
#    labels to classify the provided data.
#
if False:
    classifier = pipeline("zero-shot-classification")
    pretty_print(
        classifier(
            "This is a course about the Transformers library",
            candidate_labels=["education", "politics", "business"],
        )
    )

    pretty_print(
        classifier(
            [
                "Several countries will have elections next year.",
                "OpenAI is contiously loosing key people.",
            ],
            candidate_labels=["education", "politics", "business"],
        )
    )

    pretty_print(
        classifier(
            [
                "Please solve for x.",
                "Write a summary for the given text and answer the question a to f.",
                "What is the highest mountain on earth",
                "Do another ten cylces.",
                "Explain in a short paragraph why the intial study of the prisoner's dilemma is controversial.",
                "What came first the chicken or the egg.",
            ],
            candidate_labels=[
                "math",
                "language",
                "geography",
                "physical education",
                "ethics",
            ],
        )
    )


#
# Text generation
#
if False:
    generator = pipeline("text-generation", num_return_sequences=2, max_length=16)
    pretty_print(generator("In this course, we will teach you how to"))

if False:
    generator = pipeline("text-generation", model="distilgpt2")
    pretty_print(
        generator(
            "In this course, we will teach you how to",
            max_length=30,
            num_return_sequences=2,
        )
    )


#
# Mask filling
#
if False:
    unmasker = pipeline("fill-mask")
    pretty_print(
        unmasker("This course will teach you all about <mask> models.", top_k=2)
    )

    unmasker = pipeline("fill-mask", model="bert-base-cased")
    pretty_print(
        unmasker("This course will teach you all about [MASK] models.", top_k=2)
    )

#
# Name entity recognition (NER)
#
if False:
    ner = pipeline("ner", grouped_entities=True)

    pretty_print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

    ner = pipeline(
        "ner",
        model="vblagoje/bert-english-uncased-finetuned-pos",
        grouped_entities=True,
    )

    pretty_print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))


#
# Question Answering
#
if False:
    question_answerer = pipeline("question-answering")
    pretty_print(
        question_answerer(
            question="Where do I work?",
            context="My name is Sylvain and I work at Hugging Face in Brooklyn",
        )
    )

#
# Summarization
#
if False:
    summarizer = pipeline("summarization")
    pretty_print(
        summarizer(
            """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """
        )
    )

#
# Translation
#
if True:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    pretty_print(translator("Ce cours est produit par Hugging Face."))
