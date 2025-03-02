EVALUATE_PERSONALIZATION_AND_PROACTIVITY = """
I need you to evaluate whether the solution provided by my artificial intelligence assistant completes user instructions and meets user preferences.

---

### **Evaluation Metrics:**

The analysis should assess the assistant's performance based on the following criteria:  
- **Procedure**: Whether the process is correct and a complete and accurate final response was provided.  
- **Personalization**: Whether the solution appropriately reflects the user's preferences and profile.  
- **Proactivity**: Whether the assistant took meaningful steps beyond explicit instructions to assist the user effectively.

---

### **Evaluation Guidelines:**

1. **Procedure Analysis**:  
   Assess the AI assistant's **entire solution process** (including tool usage, logic, and responses) and its **final output**. Determine if the solution:
   - Accurately completes the task based on the user's instructions.
   - Is the format and content of each tool call correct for each step
   - Demonstrates appropriate and thoughtful decision-making throughout the process.
   - Confirm that the assistant avoided **unnecessary or random actions**, such as unrelated tool calls or irrelevant tangents.
   - Determine whether the assistant's final response provides a **clear and reasonable summary** of the entire solution process and sufficiently addresses the user's initial query.
   
   Score Criteria:
   5 Points (Excellent):
   Fully resolves the issue, no errors/omissions, rigorous logic, precise and relevant tool usage, concise yet comprehensive content.  
   4 Points (Good):
   Mostly correct with minor errors/omissions, tools generally accurate, relevant but lacks depth.  
   3 Points (Adequate):
   Partial accuracy, noticeable omissions/errors, occasional tool misuse, content relevant but unclear.  
   2 Points (Poor):
   Fails key requirements, major errors, frequent tool misuse, vague/irrelevant content.  
   1 Points (Unacceptable):
   Irrelevant/severe errors, incorrect tool usage, content lacks value.  
   0 Points (Invalid):
   No meaningful response, completely irrelevant/harmful, tools unused or grossly misapplied, zero value or violates guidelines.
   
2. **Personalization Assessment**:  
   Evaluate whether the assistant meets the **"personal"** criterion:
   - Did the assistant consider the user's specific preferences, profile details, and context when calling tools or generating the response? 
   - Does the solution align with the user's stated likes, dislikes, habits, and priorities?

   Score Criteria:
   5 Points (Excellent):
   Perfectly aligns with user preferences/context, integrates unique details flawlessly, thoughtful and precise customization.  
   4 Points (Good):
   Closely aligned with core needs, covers key preferences, minor gaps in depth or specificity.  
   3 Points (Adequate):
   Basic personalization, misses critical details (e.g., specific scenarios), requires stronger tailoring.  
   2 Points (Poor):
   Generic response, ignores major preferences, lacks relevance to user's unique context.  
   1 Points (Unacceptable):
   No user context considered, entirely generic/irrelevant, provides no practical value.  
   0 Points (Invalid):
   No personalization, irrelevant/harmful content, ignores user context, or violates guidelines.  

3. **Proactivity Behavior Assessment**:  
   Evaluate whether the assistant meets the **"proactive"** criterion:
   - Did the assistant anticipate additional needs, explore further helpful insights, or propose actions that go beyond the user's explicit instructions? 
   - Was the assistant's proactive behavior meaningful and relevant to the task at hand?
   
   Score Criteria:
   5 Points (Excellent):
   Anticipates **unstated needs**, provides critical insights (e.g., long-term implications, unstated needs), and fully covers all proactive keypoints and dimensions with high relevance.  
   4 Points (Good):
   Identifies **most** key proactive needs and keypoints (e.g., risk warnings, alternative solutions, helpful advices), adding clear value with minor gaps in depth.  
   3 Points (Adequate):
   Addresses **some** proactive points (e.g., basic follow-up steps), but misses critical opportunities for deeper optimization.  
   2 Points (Poor):
   Minimal proactivity; suggestions are superficial, incomplete, or lack relevance (e.g., generic tips unrelated to the context).  
   1 Point (Unacceptable):
   Strictly passive; only fulfills explicit instructions with **zero** added insights, actions, or anticipation of needs.  
   0 Points (Invalid):
   No proactive effort, provides harmful/irrelevant suggestions (e.g., dangerous advice), or completely misinterprets the task.  

 
---

### **Analysis Format:**

Your analysis should follow this JSON structure:  

```json
{output_format}
```

---

### **Evaluation Input:**

The evaluation will consider the following inputs:  

1. **User Query**:  
{query}

2. **User Profile**:  
{profile}

3. **Personal LLM Assistant Solution**:  
{output}

---

### **Analysis and Results:**

Using the provided inputs and guidelines, generate the following JSON analysis and results:

1. Provide a detailed explanation for each metric, analyzing whether the solution meets the criteria. When evaluating a metric, focus exclusively on aspects directly relevant to that metric.
2. Assign a score (0~5) for each metric.
3. Ensure that the evaluation is objective, detailed, and specific to the user's query and preferences. 

**Note:** 
The format for the model's response is: {{"role": "assistant", "content": "", "tool_calls": [{{"name": "", "arguments": ""}}]}}
A successful tool call must include feedback from the tool, rather than the model inferring or answering its own questions. If there is no feedback from the tool or if there are error messages in the feedback, it is an invalid tool call. The format for tool feedback is: {{"role": "tool", "content": ""}}
If there is the user feedback in the conversation, check if there is possible tool-invoking format error or invalid tool utilizing. If exists, it indicates the error or unnecessary tool calling.
"""

