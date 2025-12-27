{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ddca40e0",
   "metadata": {},
   "source": [
    "# Analyze and Reason on Multimodal Data with Gemini: Challenge Lab"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e4f6409",
   "metadata": {},
   "source": [
    "## GSP524"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a3362c8",
   "metadata": {},
   "source": [
    "## Challenge Scenario\n",
    "\n",
    "#### Cymbal Direct: Analyzing Social Media Engagement for a New Product Launch\n",
    "\n",
    "Cymbal Direct just launched a new line of athletic apparel designed for enhanced performance during various activities. To gauge public perception and potential market impact, Cymbal Direct is tasked with analyzing social media engagement across multiple platforms. This analysis will involve:\n",
    "  * **Text**: Analyzing customer reviews and social media posts for sentiment and key themes.\n",
    "  * **Image**: Analyzing images posted by influencers and customers wearing the apparel to identify style trends and usage patterns.\n",
    "  * **Audio** Analyzing an audio clip of a podcast episode of a recent interview about Cymbal Direct's new product launch.\n",
    "\n",
    "The goal is to provide Cymbal Direct with actionable insights to refine their marketing strategy, improve their products, and bolster product positioning. Are you ready for the challenge?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7dc93dd9",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Task 1. Import libraries and install the Gen AI SDK\n",
    "\n",
    "In this section, you will import the libraries required for this lab and install the Google Gen AI SDK.\n",
    "\n",
    "**All cells have been written for you in this section. There are no `#TODOs` required.**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2afe421-c806-49b5-8d23-a26837c1726c",
   "metadata": {},
   "source": [
    "### Install Google Gen AI SDK for Python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "296c092e-b172-4cad-8d5f-1d3b30af527d",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "%pip install --upgrade --quiet google-genai"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a921ab87-6781-4968-9c32-07e9eb6c6d32",
   "metadata": {},
   "source": [
    "### Restart current runtime\n",
    "\n",
    "To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8738de3f-5e9d-420d-95ae-3303405c7c8d",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Restart kernel after installs so that your environment can access the new packages\n",
    "import IPython\n",
    "\n",
    "app = IPython.Application.instance()\n",
    "app.kernel.do_shutdown(True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "208e9e92-db5a-42b7-84db-accebae25362",
   "metadata": {},
   "source": [
    "### Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0427595f-affe-4de0-99db-2997b65c4901",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from IPython.display import HTML, Markdown, display\n",
    "from google import genai\n",
    "from google.genai import types\n",
    "from google.genai.types import (\n",
    "    FunctionDeclaration,\n",
    "    GenerateContentConfig,\n",
    "    GoogleSearch,\n",
    "    MediaResolution,\n",
    "    Part,\n",
    "    Retrieval,\n",
    "    SafetySetting,\n",
    "    Tool,\n",
    "    ToolCodeExecution,\n",
    "    ThinkingConfig,\n",
    "    GenerateContentResponse,\n",
    "    GenerateContentConfig,    \n",
    "    VertexAISearch,\n",
    ")\n",
    "from collections.abc import Iterator\n",
    "import os"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c061bbe1-2c1c-4d31-9bfb-1afc6d9d3f9e",
   "metadata": {},
   "source": [
    "### Set Google Cloud project information and initialize Google Gen AI SDK"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce95f701-9639-40cf-8736-3a6fbe3fe8a1",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "os.makedirs('analysis', exist_ok=True)\n",
    "\n",
    "PROJECT_ID = \"qwiklabs-gcp-03-832f2fbae9a3\"\n",
    "LOCATION = os.environ.get(\"GOOGLE_CLOUD_REGION\", \"global\")\n",
    "print(f\"Project ID: {PROJECT_ID}\")\n",
    "print(f\"LOCATION: {LOCATION}\")\n",
    "\n",
    "client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f44701ac-5bcc-4b07-bbc8-23bc4d3289ba",
   "metadata": {},
   "source": [
    "### Load the Gemini 2.5 Flash model\n",
    "\n",
    "Learn more about all [Gemini models on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a67c028a-30e0-4c46-88a6-8a4eb3021cd1",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "MODEL_ID = \"gemini-2.5-flash\"  # @param {type: \"string\"}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ea28075",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Task 2. Analyze and reason on customer feedback (text)\n",
    "\n",
    "In this task, you'll use the Gemini 2.5 Flash model to analyze customer reviews and social media posts in text format about Cymbal Direct's new athletic apparel. You will save the findings from the model into a markdown file that you will use for a comprehensive report in the last task.\n",
    "\n",
    "**Your tasks will be labeled with a `#TODO` section in the cell. Read each cell carefully and ensure you are filling them out correctly!**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9c47c18b-d8f6-49f5-9ec7-1914392213fd",
   "metadata": {
    "tags": []
   },
   "source": [
    "###  Load and preview the text data\n",
    "This file contains customer reviews and social media posts about Cymbal Direct's new athletic apparel line, collected from various e-commerce platforms and social media sites. The data is in raw text format, with each review or post separated by a newline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4897132-d588-4bf1-818c-ddcdb0f67610",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load and preview the text data (reviews.txt)\n",
    "!gcloud storage cp gs://{PROJECT_ID}-bucket/media/text/reviews.txt media/text/reviews.txt\n",
    "!head media/text/reviews.txt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8ad5584d-1481-4004-8744-e58ebb9e4581",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Initial Analysis with Gemini 2.5 Flash\n",
    "For this section, you will need to fill out the `#TODOs` for **Construct the prompt for Gemini** and **Send the prompt to Gemini**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7715e2b5-637e-41c3-a4bb-7274baa9469c",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# 1. Load the text data (reviews.txt)\n",
    "with open('media/text/reviews.txt', 'r') as f:\n",
    "    text_data = f.read()\n",
    "\n",
    "# 2. Construct the prompt for Gemini\n",
    "# TODO: Write a prompt that instructs the Gemini model to analyze the customer reviews and social media posts.\n",
    "# The prompt should include clear instructions to:\n",
    "# - Identify the overall sentiment (positive, negative, or neutral) of each review or post.\n",
    "# - Extract key themes and topics discussed, such as product quality, fit, style, customer service, and pricing.\n",
    "# - Identify any frequently mentioned product names or specific features.\n",
    "prompt = f\"\"\"\n",
    "[Your prompt here]\n",
    "{text_data}\n",
    "\"\"\"\n",
    "\n",
    "# 3. Send the prompt to Gemini\n",
    "# TODO: Use the `client.models.generate_content` method to send the prompt and text data to the Gemini model.\n",
    "# TODO: Make sure to specify the `MODEL_ID` and the `prompt` as parameters.\n",
    "# TODO: Store the response from the model in a variable named `response`.\n",
    "\n",
    "# 4. Display the response\n",
    "display(Markdown(response.text))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a55c2adc-cdd2-4c81-bab5-28fd28bceb6e",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Deep Dive with Gemini 2.5 Flash Model  \n",
    "\n",
    "Now that you have generated some insights based on the reviews, you will use the Gemini 2.5 Flash model to explore the reviews in more detail, and come up with some takeaways and use reasoning to create actionable insights for your team."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9918b624-4e00-4ebb-8967-f0029c8b26ab",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "MODEL_ID = \"gemini-2.5-flash\"  # @param {type: \"string\"}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "827fd66b-561e-4cba-a8d3-33fd4b74e490",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Helper functions\n",
    "\n",
    "Create methods to print out the thoughts and answer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e05e0640-05ef-4d25-bd13-9f4bc0a6c3d8",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def print_thoughts(response: GenerateContentResponse) -> None:\n",
    "    for part in response.candidates[0].content.parts:\n",
    "        header = \"Thoughts\" if part.thought else \"Answer\"\n",
    "        display(Markdown(f\"\"\"## {header}:\\n{part.text}\"\"\"))\n",
    "\n",
    "\n",
    "def print_thoughts_stream(response: Iterator[GenerateContentResponse]) -> None:\n",
    "    display(Markdown(\"## Thoughts:\\n\"))\n",
    "    answer_shown = False\n",
    "\n",
    "    for chunk in response:\n",
    "        for part in chunk.candidates[0].content.parts:\n",
    "            if not part.thought and not answer_shown:\n",
    "                display(Markdown(\"## Answer:\\n\"))\n",
    "                answer_shown = True\n",
    "            display(Markdown(part.text))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1252023e-cdf0-4d06-8f5a-a16e0bd1cc5b",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Enable thoughts\n",
    "\n",
    "You set the flag `include_thoughts` in the `ThinkingConfig` to indicate whether to return thoughts in the model response. The flag is set to `False` by default. You will also set the optional `thinking_budget` parameter in the ThinkingConfig to control and configure how much a model thinks on a given user prompt.\n",
    "\n",
    "The `thinkingBudget` parameter guides the model on the number of thinking tokens to use when generating a response. A higher token count generally allows for more detailed reasoning, which can be beneficial for tackling more complex tasks. If latency is more important, use a lower budget or disable thinking by setting thinkingBudget to `0`. Setting the thinkingBudget to `-1` turns on dynamic thinking, meaning the model will adjust the budget based on the complexity of the request. For the purposes of this lab, dynamic thinking has been enabled in the following code block. You will use this config for enabling Gemini thinking. For more information, check out the following [documentation](https://ai.google.dev/gemini-api/docs/thinking).\n",
    "\n",
    "\n",
    "**Hint: you will need to use this for calls that require extra reasoning.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f3737eb-b8e6-4e52-822e-f9cf2ba66619",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(include_thoughts=True,thinking_budget=-1))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0723a30-0a7f-4820-bb76-e055a02509cc",
   "metadata": {
    "tags": []
   },
   "source": [
    "### Deep Dive with Gemini 2.5 Flash: Reasoning on Customer Sentiment\n",
    "\n",
    "In this section, you'll use Gemini thinking to delve deeper into the customer sentiment and identify key areas for improvement. We're particularly interested in understanding the reasoning behind positive and negative reviews and uncovering any recurring themes that might not be immediately apparent.\n",
    "\n",
    "For this section, you will need to fill out the `#TODOs` for **Construct the prompt for Gemini** and **Use Gemini Thinking for deeper reasoning**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48bf95b6-dfe8-4258-8f22-ecba67530661",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# 1. Construct the prompt for Gemini\n",
    "# TODO: Write a prompt that instructs the Gemini model to analyze the customer reviews and social media posts in more detail.\n",
    "# The prompt should include clear instructions to:\n",
    "# - Identify the main factors driving positive and negative sentiment.\n",
    "# - Assess the overall impact of the new athletic apparel line on brand perception.\n",
    "# - Identify three key areas where Cymbal Direct can improve customer satisfaction or product offerings.\n",
    "# - Imagine you are presenting your findings to the Cymbal Direct marketing team and highlight the three most important takeaways.\n",
    "thinking_mode_prompt = f\"\"\"\n",
    "[Your prompt here]\n",
    "{text_data}\n",
    "\"\"\"\n",
    "\n",
    "# 2. Use Gemini thinking for deeper reasoning\n",
    "# TODO: Use the `client.models.generate_content` method to send the prompt and text data to the Gemini model.\n",
    "# TODO: Make sure to specify the `MODEL_ID` and the `thinking_mode_prompt` as parameters.\n",
    "# TODO: Also, pass the `config` object to enable thinking mode.\n",
    "# TODO: Store the response from the model in a variable named `thinking_model_response`.\n",
    "\n",
    "# 3. Print thoughts and answer\n",
    "print_thoughts(thinking_model_response)\n",
    "\n",
    "# 4. Save the text analysis to a file\n",
    "with open('analysis/text_analysis.md', 'w') as f:\n",
    "    f.write(thinking_model_response.text)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "27eb9ee1",
   "metadata": {},
   "source": [
    "## Task 3. Analyze and reason on visual content: Style trends and customer behavior\n",
    "\n",
    "In this task, you'll focus on analyzing images related to Cymbal Direct's new athletic apparel line. The goal is to identify style trends and customer behavior based on the images. You will save the findings from the model into a markdown file that you will use for a comprehensive report in the last task.\n",
    "\n",
    "**Your tasks will be labeled with a `#TODO` section in the cell. Read each cell carefully and ensure you are filling them out correctly!**\n",
    "\n",
    "#### Introduction and Context\n",
    "This image dataset consists of a mix of product photos and influencer posts showcasing Cymbal Direct's new athletic apparel line. The images feature models and influencers wearing the apparel in various settings, providing visual information about style, usage patterns, and target audience."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5df874ac-afa7-42e8-9883-1c540ec35194",
   "metadata": {},
   "source": [
    "### Load and preview the image data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fffb0a43-075e-4a85-acdd-bb696a893ac7",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "!gcloud storage cp -r gs://{PROJECT_ID}-bucket/media/images media/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4e4255ca-7759-4389-9130-86dc14761f00",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "\n",
    "# Specify the directory where the images are stored\n",
    "image_dir = 'media/images'\n",
    "\n",
    "# Get a list of all image files in the directory\n",
    "image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]\n",
    "\n",
    "# Display the images\n",
    "for image_file in image_files:\n",
    "    # Construct the full path to the image file\n",
    "    image_path = os.path.join(image_dir, image_file)\n",
    "    \n",
    "    # Load the image\n",
    "    img = mpimg.imread(image_path)\n",
    "    \n",
    "    # Display the image using Matplotlib\n",
    "    plt.figure()  # Create a new figure for each image\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')  # Hide the axis\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d83edefe-5311-407d-8d0c-19b7c291241d",
   "metadata": {},
   "source": [
    "### Initial Analysis with Gemini 2.5 Flash\n",
    "\n",
    "For this section, you will need to fill out the `#TODOs` for **Construct the prompt for Gemini** and **Send the prompt and images to Gemini**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94258f26-4f50-4ade-9ed8-434c1c6b3919",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "MODEL_ID = \"gemini-2.5-flash\"  # @param {type: \"string\"}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "300fecb4-551c-4ff3-83b5-a82adc0a344f",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# 1. Load the image data\n",
    "image_folder = 'media/images'\n",
    "image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]\n",
    "\n",
    "# 2. Load the images into a list of `Part` objects\n",
    "image_parts = []\n",
    "for image_file in image_files:\n",
    "    image_path = os.path.join(image_folder, image_file)\n",
    "    with open(image_path, 'rb') as f:\n",
    "        image_bytes = f.read()\n",
    "    image_parts.append(Part.from_bytes(data=image_bytes, mime_type='image/jpeg'))  # Adjust mime_type if needed\n",
    "\n",
    "# 3. Construct the prompt for Gemini\n",
    "# TODO: Write a prompt that instructs the Gemini model to analyze the images of Cymbal Direct's new athletic apparel line.\n",
    "# The prompt should include clear instructions to:\n",
    "# - Identify the apparel items in each image.\n",
    "# - Describe the attributes of each item.\n",
    "# - Identify any prominent style trends or preferences.\n",
    "prompt = f\"\"\"\n",
    "[Your prompt here]\n",
    "\"\"\"  # TODO: Add your prompt here\n",
    "\n",
    "# 4. Send the prompt and images to Gemini\n",
    "# TODO: Use the `client.models.generate_content` method to send the prompt and images to the Gemini model.\n",
    "# TODO: Make sure to specify the `MODEL_ID` and the `contents` (including the prompt and image parts) as parameters.\n",
    "# TODO: Store the response from the model in a variable named `response`.\n",
    "\n",
    "# 5. Display the response\n",
    "display(Markdown(response.text))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0bd3ae7-adf5-49b8-a4f5-386cfd85a1e6",
   "metadata": {},
   "source": [
    "### Reasoning on image trends with Gemini 2.5 Flash\n",
    "\n",
    "You'll now use Gemini thinking to perform a more in-depth analysis of the visual elements, inferring context, target audience, and potential marketing implications.\n",
    "\n",
    "For this section, you will need to fill out the `#TODOs` for **Construct the prompt for Gemini** and **Use Gemini Thinking for deeper reasoning**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8af68c30-4e17-4eca-8f05-15805beac3fb",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "MODEL_ID = \"gemini-2.5-flash\"  # @param {type: \"string\"}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3ae688d-c8c0-457e-baab-e1699daa214a",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# 1. Construct the prompt for Gemini\n",
    "# TODO: Write a prompt that instructs the Gemini model to analyze the images in more detail.\n",
    "# The prompt should include clear instructions to:\n",
    "# - Develop a hypothesis about the target audience for each image.\n",
    "# - Analyze how visual elements contribute to the overall message and appeal.\n",
    "# - Compare the observed style trends with broader fashion trends in athletic wear.\n",
    "# - Provide recommendations for Cymbal Direct's future marketing campaigns or product development.\n",
    "thinking_mode_prompt = f\"\"\"\n",
    "[Your prompt here]\n",
    "\"\"\"\n",
    "\n",
    "# 2. Use Gemini thinking for deeper reasoning\n",
    "# TODO: Use the `client.models.generate_content` method to send the thinking_mode_prompt and images to the Gemini model.\n",
    "# TODO: Make sure to specify the `MODEL_ID`, `contents` (including the prompt and image parts), and `config` to enable thinking mode.\n",
    "# TODO: Store the response from the model in a variable named `thinking_model_response_image`.\n",
    "\n",
    "# 3. Print thoughts and answer\n",
    "print_thoughts(thinking_model_response_image)\n",
    "\n",
    "# 4. Save the image analysis to a file\n",
    "with open('analysis/image_analysis.md', 'w') as f:\n",
    "    f.write(thinking_model_response_image.text)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c5e9312",
   "metadata": {},
   "source": [
    "## Task 4. Analyze and reason on audio content: Customer perceptions\n",
    "\n",
    "In this section, you will use Gemini to analyze a podcast about Cymbal Direct's new clothing line and extract information/sentiment out of it and use those to generate insights for the company. You will save the findings from the model into a markdown file that you will use for a comprehensive report in the last task.\n",
    "\n",
    "**Your tasks will be labeled with a `#TODO` section in the cell. Read each cell carefully and ensure you are filling them out correctly!**\n",
    "\n",
    "#### Introduction and Context\n",
    "This audio clip is from a podcast episode featuring an interview with a Cymbal Direct representative discussing the new athletic apparel line. The conversation covers various aspects of the apparel, including design, features, target audience, and marketing strategy."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7429a162-3421-44b0-9c6d-97c00b14ce6d",
   "metadata": {},
   "source": [
    "### Preview the podcast episode (optional)\n",
    "\n",
    "To listen to the podcast episode, you can copy the file to your local environment and use iPython to preview it in the notebook."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f5f7ac6e-9628-4375-83eb-386e834f20eb",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import IPython\n",
    "\n",
    "!gcloud storage cp gs://{PROJECT_ID}-bucket/media/audio/cymbal_direct_expert_interview.wav \\\n",
    "media/audio/cymbal_direct_expert_interview.wav\n",
    "\n",
    "IPython.display.Audio('media/audio/cymbal_direct_expert_interview.wav')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4371f37f-3716-4048-b5b4-1f53a9f73eeb",
   "metadata": {},
   "source": [
    "### Initial analysis with Gemini 2.5 Flash\n",
    "For this section, you will need to fill out the `#TODOs` for **Construct the prompt for Gemini** and **Send the prompt and audio to Gemini**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "48c77803-453b-4d50-96a6-ce85ffe7a94a",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "MODEL_ID = \"gemini-2.5-flash\"  # @param {type: \"string\"}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae0c1641-9196-4a7f-8544-3522cfb480d9",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Construct the file URI using f-string\n",
    "file_uri = f\"gs://{PROJECT_ID}-bucket/media/audio/cymbal_direct_expert_interview.wav\"\n",
    "\n",
    "audio_part = Part.from_uri(\n",
    "    file_uri=file_uri,\n",
    "    mime_type=\"audio/wav\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0dda09fa-d27d-4553-b7b9-3586dd6d8feb",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# 1. Construct the prompt for Gemini\n",
    "# TODO: Write a prompt that instructs the Gemini model to analyze the audio recording of the conversation about Cymbal Direct's new athletic apparel line.\n",
    "# The prompt should include clear instructions to:\n",
    "# - Transcribe the conversation, identifying different speakers.\n",
    "# - Provide a sentiment analysis, highlighting positive, negative, and neutral opinions.\n",
    "# - Identify key themes and topics discussed, such as comfort, fit, performance, style, and comparisons to competitors.\n",
    "prompt = f\"\"\"\n",
    "[Your prompt here]\n",
    "\"\"\"\n",
    "\n",
    "# 2. Send the prompt and audio to Gemini\n",
    "# TODO: Use the `client.models.generate_content` method to send the thinking_mode_prompt and audio data to the Gemini model.\n",
    "# TODO: Make sure to specify the `MODEL_ID` and the `contents` (including the `audio_part` and the `prompt`) as parameters.\n",
    "# TODO: Store the response from the model in a variable named `response`.\n",
    "\n",
    "# 3. Display the response\n",
    "display(Markdown(response.text))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4999e48f-a499-41dd-89cb-2f345ceda39c",
   "metadata": {},
   "source": [
    "### Reasoning on Audio Insights with Gemini 2.5 Flash\n",
    "In this section, you'll use Gemini thinking to analyze the conversation at a deeper level, reason about customer satisfaction, deduce influencing factors, and generate data-driven recommendations.\n",
    "\n",
    "For this section, you will need to fill out the `#TODOs` for **Construct the prompt for Gemini** and **Use Gemini Thinking for deeper reasoning**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0370afbe-97fe-4396-85c0-b07b21d5f0ce",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "MODEL_ID = \"gemini-2.5-flash\"  # @param {type: \"string\"}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3479aaf5-f2e7-4b5c-9073-f48788dcbcbe",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# 1. Construct the prompt for Gemini\n",
    "# TODO: Write a prompt that instructs the Gemini model to analyze the audio recording in more detail.\n",
    "# The prompt should include clear instructions to:\n",
    "# - Reason about the overall customer satisfaction with the apparel.\n",
    "# - Deduce the key factors influencing customer perception.\n",
    "# - Develop three data-driven recommendations for Cymbal Direct.\n",
    "# - Identify any potential biases or limitations in the audio data.\n",
    "thinking_mode_prompt = \"\"\"\n",
    "[Your prompt here]\n",
    "\"\"\"\n",
    "\n",
    "# 2. Use Gemini thinking for deeper reasoning\n",
    "# TODO: Use the `client.models.generate_content` method to send the prompt and audio data to the Gemini model.\n",
    "# TODO: Make sure to specify the `MODEL_ID`, `contents` (including the `audio_part` and the `prompt`), and `config` to enable thinking mode.\n",
    "# TODO: Store the response from the model in a variable named `thinking_model_response`.\n",
    "\n",
    "# 3. Print the thoughts and answer\n",
    "print_thoughts(thinking_model_response)\n",
    "\n",
    "# 4. Save the audio analysis to a text file in the analysis folder\n",
    "with open('analysis/audio_analysis.md', 'w') as f:\n",
    "    f.write(thinking_model_response.text)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05c4778d-bd3f-4678-91ff-eecee571b521",
   "metadata": {},
   "source": [
    "## Task 5. Synthesize multimodal insights: Generate a comprehensive report\n",
    "\n",
    "In this final task, you will synthesize the insights gained from your previous analyses of text, images, and audio data. You'll use the Gemini 2.5 Flash model to generate a comprehensive report that consolidates the findings from each modality, providing a holistic view of customer sentiment, style preferences, and key trends related to Cymbal Direct's new athletic apparel line.\n",
    "\n",
    "You will save the final report generated by the model into a markdown file, which you will then upload to Cloud Storage for review and evaluation. This comprehensive report will serve as a valuable resource for Cymbal Direct, enabling them to make informed decisions and optimize their strategies based on a thorough understanding of customer perceptions and market trends.\n",
    "\n",
    "**Your tasks will be labeled with a #TODO section in the cell. Read each cell carefully and ensure you are filling them out correctly!**\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d953b50-a1b7-402c-8e40-1c5d4e16fac3",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "MODEL_ID = \"gemini-2.5-flash\"  # @param {type: \"string\"}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b8fe95a-dd64-4a1f-8dcf-1b7183f26de8",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# 1. Load the analysis results from the files\n",
    "with open('analysis/text_analysis.md', 'r') as f:\n",
    "    text_analysis = f.read()\n",
    "\n",
    "with open('analysis/image_analysis.md', 'r') as f:\n",
    "    image_analysis = f.read()\n",
    "\n",
    "with open('analysis/audio_analysis.md', 'r') as f:\n",
    "    audio_analysis = f.read()\n",
    "\n",
    "# 2. Combine the analysis results\n",
    "all_analysis = f\"\"\"\n",
    "## Text Analysis:\n",
    "{text_analysis}\n",
    "\n",
    "## Image Analysis:\n",
    "{image_analysis}\n",
    "\n",
    "## Audio Analysis:\n",
    "{audio_analysis}\n",
    "\"\"\"\n",
    "\n",
    "# 3. Construct the prompt for Gemini\n",
    "# TODO: Write a prompt to instruct the Gemini model to generate a comprehensive report based on the combined analysis results.\n",
    "# The prompt should include clear instructions to:\n",
    "# - Summarize the overall sentiment towards the new apparel line.\n",
    "# - Identify key themes and trends in customer feedback.\n",
    "# - Provide insights on style preferences, usage patterns, and customer behavior.\n",
    "# - Evaluate the audio and its fit with the product image.\n",
    "# - Offer actionable recommendations for Cymbal Direct to refine their marketing strategy and product positioning.\n",
    "comprehensive_report_prompt = f\"\"\"\n",
    "[Your prompt here]\n",
    "{all_analysis}\n",
    "\"\"\"\n",
    "\n",
    "# 4. Send the prompt to Gemini\n",
    "# TODO: Use the `client.models.generate_content` method to send the comprehensive_report_prompt to the Gemini model.\n",
    "# TODO: Make sure to specify the `MODEL_ID`, the `comprehensive_report_prompt`, and the `config` to enable thinking mode.\n",
    "# TODO: Store the response from the model in a variable named `thinking_model_response`.\n",
    "\n",
    "# 5. Print the thoughts and answer\n",
    "print_thoughts(thinking_model_response)\n",
    "\n",
    "# 6. Save the final report to a file\n",
    "with open('analysis/final_report.md', 'w') as f:\n",
    "    f.write(thinking_model_response.text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c594e36b-9cce-4536-a143-c6e709d0b717",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "!gcloud storage cp analysis/final_report.md gs://{PROJECT_ID}-bucket/analysis/final_report.md"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dcc1497b-bb2c-4eb0-9a55-02c700a3cf30",
   "metadata": {},
   "source": [
    "## Congratulations!\n",
    "\n",
    "Congratulations! In this lab, you have successfully utilized the Gemini 2.5 Flash model with reasoning capabilities to analyze multimodal data, including text, images, and audio, to gain valuable insights for Cymbal Direct's new athletic apparel line. You have demonstrated proficiency in constructing effective prompts, leveraging the reasoning capabilities of Gemini thinking, and generating a comprehensive report with actionable recommendations."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "95d2f769-7990-498f-a5d5-04e08bfa29a9",
   "metadata": {},
   "source": [
    "Copyright 2025 Google LLC All rights reserved. Google and the Google logo are trademarks of Google LLC. All other company and product names may be trademarks of the respective companies with which they are associated."
   ]
  }
 ],
 "metadata": {
  "environment": {
   "kernel": "conda-base-py",
   "name": "workbench-notebooks.m128",
   "type": "gcloud",
   "uri": "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/workbench-notebooks:m128"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel) (Local) (Local)",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
