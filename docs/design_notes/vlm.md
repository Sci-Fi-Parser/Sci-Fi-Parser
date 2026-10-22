# VLM

VLM stage of the pipeline requires an endpoint using OpenAI Chat Completions API.

We used local VLM models through Ollama and llama.cpp, so the program has been tested using those two runners, but should work for any LLM server that provides a Chat Completions endpoint.

We decided to not have the program run the VLM itself, to keep swapping models easy and lightweight.

VLM output is constrained by a pydantic schema which is sent in the request's response_format field to the VLM runner.

The schema does not allow for thinking before the JSON-formatted output, but implementing this shouldn't be too hard. 
