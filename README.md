# Sentiment Analysis of PlanRVA Public Comments using BART-large-mnli

This project utilizes Natural Language Processing (NLP) techniques to perform sentiment analysis on public review comments associated with PlanRVA's Public Engagement and Outreach Report, specifically related to the development of the ConnectRVA 2045 Long-Range Transportation Plan.

## Project Overview

In 2020, the Richmond Regional Transportation Planning Organization initiated an update to the regional Long-Range Transportation Plan, branded as ConnectRVA 2045. An integral part of this process involved public engagement, where community members provided feedback on various proposed transportation projects. This project aims to classify these public comments into sentiment categories: supportive, opposed, and skeptical, employing advanced NLP techniques.

## Data Source

The comments analyzed in this project were extracted from the publicly available ConnectRVA 2045 Public Engagement & Outreach Report. This report encapsulates the community's feedback throughout the planning process, offering a rich dataset for understanding public sentiment towards proposed transportation initiatives in the Richmond, VA region.

## Methodology

### NLP and Zero-Shot Classification

This project leverages the BART-large-mnli model, a powerful transformer-based model known for its effectiveness in Natural Language Understanding (NLU) tasks. The model is utilized here for zero-shot text classification, a technique allowing the classification of text into predefined categories without the need for labeled training data.

### Analysis Steps

1. **Data Extraction and Preprocessing**: The public comments are extracted from the PDF report and preprocessed for analysis.
2. **Zero-Shot Classification**: Using the BART-large-mnli model, each comment is classified into one of three sentiment categories without the need for explicit training on sentiment-labeled data.
3. **Aggregation and Visualization**: The classified sentiments are aggregated and visualized to reflect the community's overall sentiment towards the proposed projects.

## Results

The sentiment analysis reveals the community's general attitudes towards the proposed transportation projects. By categorizing the public's responses, we can discern which projects garnered the most support, faced opposition, or were met with skepticism. These insights can aid stakeholders in making informed decisions and addressing community concerns.

## Discussion

The application of NLP and zero-shot classification in analyzing public sentiment provides a novel approach to understanding large volumes of unstructured text data. This project demonstrates the potential of these techniques in extracting meaningful insights from public engagement efforts, which can be crucial for policy-making and planning processes.

## How to Use

Instructions for replicating the analysis and exploring the results are provided. Users are encouraged to review the methodology section and utilize the provided code and datasets to conduct their own analysis.

## Acknowledgements

Special thanks to all the organizations and individuals involved in the ConnectRVA 2045 planning process for making the Public Engagement & Outreach Report publicly available, enabling this analysis.

## Plots

![image](https://github.com/planwithdata/Sentiment-Analysis_NLP/assets/131815755/e12fb56f-c6b0-45ac-96c2-31ad97ffc505)
![image](https://github.com/planwithdata/Sentiment-Analysis_NLP/assets/131815755/a826de33-729c-45fe-9a1d-6acce8f2e119)


