---
layout: post
title: "Interpretability vs Explainability in Cheminformatics"
date: 2025-11-09 17:00:00 +1300
categories:
  - AI
  - cheminformatics
  - data science
  - machine learning
---

Interpretability and explainability are different concepts in machine learning, yet many cheminformatics authors use the terms interchangably.

---

Something I've noticed recently in cheminformatics papers is authors using the terms **interpretable** and **explainable** interchangably. These terms have clear definitions, and misusing them risks confusing readers and not communicating research efficiently. This may sound like I'm just being pedantic, but for cheminformatics to become a mature field of research alongside machine learning, we need to demand high standards and statistical rigour in our research and communication.

To illustrate this, I've selected a few quotes from a paper I read recently, [Improving Machine Learning Classification Predictions through SHAP and Features Analysis Interpretation by Pinzi, et al.](https://doi.org/10.1021/acs.jcim.5c02015) I don't want to single out one paper, as this is a broader problem in cheminformatics literature (see [here](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00519-x), [here](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00825), and [here](https://patwalters.github.io/practicalcheminformatics/jupyter/ml/interpretability/2021/06/03/interpretable.html)) and this is just one example which is an otherwise interesting paper. I also understand that for many researchers English is not a first language, which adds additional difficulties.

This first quote introduces [SHapley Additive exPlanations (SHAP)](https://shap.readthedocs.io/en/latest/):

> Besides, explainable artificial intelligence (xAI), particularly SHapley Additive exPlanations (SHAP), has gained significant attention as a powerful tool for addressing model interpretability challenges. SHAP algorithms employ cooperative game theory principles to quantify the contribution of each feature to the predictions made by machine learning models, irrespective of model complexity. In particular, SHAP values provide an objective quantification of a model’s prediction based on its input features, thereby offering clear and interpretable insights into the rationale behind model outputs.

As an example of explainable AI, SHAP values can be used to improve explainability of machine learning predictions. This is an alternative to choosing a interpretable model, but does not directly address model interpretability challenges. In this quote the authors are using *interpretable* as an adjective to describe the quality of the explanation, but when interpretable also has technical definition in machine learning, it feels imprecise to use it like this.

The paper then goes on to discuss the effect of their new method on the interpretability of machine learning predictions and machine learning models:

> This approach, tailored per cluster and cell line, can improve both the reliability and interpretability of machine learning predictions. ...this approach enhances both the efficiency of virtual screening workflows and the interpretability and robustness of machine learning-based predictive models.

Again SHAP is improving the explainability of the machine learning models and predictions. The interpretability is an intrinsic property and remains unchanged.

I can see where the confusion comes from. Interpretable has a colloquial meaning, a definition in machine learning, and can be used in cheminformatics when describing different methods for chemical representation. To fully understand the problem, the first thing to do is to define what interpretability and explainability mean in machine learning.

## Interpretability vs Explainability in Machine Learning

### Interpretability

Interpretability is *how* predictions are made. it allows humans to understand the internal mechanics of a machine learning model. The goal is transparency in the model's design, parameters, and workings. Interpretability is associated with inherently simple models like linear regression or decision trees. A simple decision tree is interpretable because you can literally follow its rules to see how it made a decision.

### Explainability

Explainability is *why* predictions are made. It refers to the ability to describe the rationale for a specific decision or prediction in human terms, often after the decision has been made. The goal is providing a justification or an understandable cause-and-effect for the output of a complex system based on the inputs. It is associated with post-hoc methods (like SHAP and LIME) when applied to complex models (often called "black-box" models like deep neural networks). These methods create an explanation for why a complex model made a specific prediction, without requiring you to understand every internal layer or parameter.

### Interpretability vs Explainability

| Aspect | Interpretability (iAI) | Explainability (xAI) |
| ---    | ---                    | ---                  |
| Core Question | **How** does the model work? | **Why** did the model make a specific prediction? |
| Focus | Understanding the model's internal logic and mechanics. | Providing a justification for the model's output. |
| Transparency | Intrinsic transparency. The model itself is transparent from the start. | Post-hoc transparency. An external method is used to shed light on the output. |
| Best Suited For | Simple models (Linear Regression, simple Decision Trees). | Complex models ("Black-Box" models: Deep Learning, Gradient Boosting Machines). |
| Scope | Typically global (understanding the model's behavior across the entire dataset). | Typically local (understanding a single prediction/instance). |
| Examples | N/A (It's a model property, not a method.) | SHAP, LIME. |

*Note: Whilst he definitions above are the most widely used and often preferred for precision, they are not universally accepted as a formal mathematical or technical definition, and there is stil some ongoing debate. Some view interpretability as a broad term to describe any effort to make a model understandable, including the results of post-hoc explanation techniques like SHAP or LIME. They see explainability as a type or measure of interpretability. The most common criticism is the lack of formal metrics for quantifying interpretability and explainability, since the terms rely on human understanding, which is subjective. There is also some debate over whether post-hoc xAI methods such as SHAP provide a genuine explanation of the model's true decision process, or merely a plausible, user-friendly rationalization or simplified approximation of it, raising questions about the trustworthiness of the explanation itself.*

## Interpretability in Chemical Representations

Interpretability is often used to describe and compare different chemical representations (embeddings) in cheminformatics. For example, structural keys such as the MACCS key are inherently interpretable, as each bit position represents a specific substructure in a molecule. Hashed fingerprints such as ECPFs are less interpretable due to bit collisions, though the specific substructures that contibute to a particular bit can be stored in a bit info map during calculation. Global calculated descriptors and learned representations from graph convolutional neural networks can be significantly less interpretable, as it is often very difficult to determine the effect of specific substructures on the values.

The confusion comes from combining an interpretable molecular representation with an explainable AI technique to understand a prediction. For example using SHAP to determine the contribution of each bit position in a MACCS key towards a machine learning prediction allows us to determine substructures that contribute to the prediction.

## How Can This Problem Be Solved?

The conflation of interpretability and explainability isn't a problem of poor intent; it stems from the desire to communicate the complex findings of black-box models effectively. However, the solution lies in rigorously adhering to the established definitions.

The fundamental distinction remains: **interpretability** is about transparency (the "white-box" model), and **explainability** is about justification (the post-hoc rationale for a "black-box" output). When using powerful models like gradient boosted decision trees or deep neural networks in drug discovery, we must accept that the model is likely not interpretable, and instead focus our reporting on the quality and fidelity of the explanation method used.

### Recommendations for Cheminformatics Researchers

To improve clarity and rigor in future work, I suggest that cheminformatics researchers should adopt the following guidelines when discussing model insights:

#### Be Precise with Terminology

Use Interpretability only when discussing models that are intrinsically understandable (e.g. a short decision tree) or when discussing a truly interpretable molecular feature representation (e.g. a structural key).

Use Explainability when discussing the application of a post-hoc method (like SHAP or LIME) to a complex black-box model.

#### Avoid Colloquial Use

Strictly avoid using the verb "interpret" or the adjective "interpretable" when referring to the output of an explanation tool like SHAP. Instead, use phrases like:

- "SHAP values explain the prediction."
- "The results provide explanations for the model's decision."
- "The analysis offers insights into feature contribution."

#### Specify the Target

Always clarify what is being explained or interpreted:

- "We used SHAP to explain the model's prediction for molecule X."
- "The MACCS key is an interpretable feature set."
- "The final model is a black box, but its predictions are explainable."

By maintaining this high standard of communication, the cheminformatics community can ensure its published work is not only scientifically sound but also clearly understood and correctly applied by the wider machine learning and pharmaceutical research fields.
