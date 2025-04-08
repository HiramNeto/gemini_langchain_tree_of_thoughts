//llm tree of thoughts.ts

// IMPORTANT - Add your API keys here
process.env.GOOGLE_API_KEY = "abcdefxxxxxxxxxxxxxxxxxxx";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";

//Defining templates for each step of the Tree of Thoughts process
const step1Template = new PromptTemplate({
  template: `Step1: I have a problem related to {input}. Could you brainstorm three distinct solutions? Please consider a variety of factors such as {perfect_factors}`,
  inputVariables: ["input", "perfect_factors"],
});

const step2Template = new PromptTemplate({
  template: `Step2: For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, initial effort needed, implementation difficulty, potential challenges, and the expected outcomes. Assign a probability of success and a confidence level to each option based on these factors {previous_solutions}`,
  inputVariables: ["previous_solutions"],
});

const step3Template = new PromptTemplate({
  template: `Step3: For each solution, deepen the thought process. Generate potential scenarios, strategies for implementation, any necessary partnerships or resources, and how potential obstacles might be overcome. Also, consider any potential unexpected outcomes and how they might be handled. {review}`,
  inputVariables: ["review"],
});

const step4Template = new PromptTemplate({
  template: `Step 4: Based on the evaluations and scenarios, rank the solutions in order of promise. Provide a justification for each ranking and offer any final thoughts or considerations for each solution {deepen_thought_process}`,
  inputVariables: ["deepen_thought_process"],
});

//Choose your LLM model
const llmModel = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",
  temperature: 1.0,
});

//Criating the chains for each step of the Tree of Thoughts
const step1Chain = RunnableSequence.from([
  (input) => {
    console.log("\n🚀 STARTING FIRST STEP: Solutions Brainstorm");
    console.log(`Problem: ${input.input}`);
    console.log(`Factors to consider: ${input.perfect_factors}`);
    return input;
  },  
  step1Template,
    llmModel,
    (output) => {
      console.log("\n✅ FIRST STEP RESULT:");
      console.log("------------------------");
      console.log(output.content);
      console.log("------------------------");
      return {
        //The output from this step is the solutions proposed by the LLM
        solutions: output.content, 
        raw_output: output 
      };
    }
  ]);
  
  const step2Chain = RunnableSequence.from([
    (input) => {
      console.log("\n🚀 STARTING SECOND STEP: pros and cons evaluation");
      console.log("Evaluating the proposed solutions...");
      return input;
    },
    RunnablePassthrough.assign({
      previous_solutions: (input) => input.solutions
    }),
    step2Template,
    llmModel,
    (output) => {
      console.log("\n✅ STEP 2 RESULT:");
      console.log("------------------------");
      console.log(output.content);
      console.log("------------------------");
      return { 
        solutions: output.solutions, 
        evaluation: output.content, 
        raw_output: output 
      };
    }
  ]);
  
  const step3Chain = RunnableSequence.from([
    (input) => {
      console.log("\n🚀 STARTING STEP 3: Deepening the thought process");
      console.log("Generating scenarios and implementation strategies...");
      return input;
    },
    RunnablePassthrough.assign({
      review: (input) => input.evaluation
    }),
    step3Template,
    llmModel,
    (output) => {
      console.log("\n✅ STEP 3 RESULT:");
      console.log("------------------------");
      console.log(output.content);
      console.log("------------------------");
      return {
        solutions: output.solutions,
        evaluation: output.evaluation,
        deepen_thought_process: output.content,
        raw_output: output
      };
    }
  ]);
  
  const step4Chain = RunnableSequence.from([
    (input) => {
      console.log("\n🚀 STARTING STEP 4: Final solution ranking");
      console.log("Ranking the solutions based on previous analyses...");
      return input;
    },
    RunnablePassthrough.assign({
      deepen_thought_process: (input) => input.deepen_thought_process,
    }),
    step4Template,
    llmModel,
    (output) => {
      console.log("\n✅ STEP 4 RESULT:");
      console.log("------------------------");
      console.log(output.content);
      console.log("------------------------");
      return {  
        final_recommendation: output.content,
        raw_output: output
      };
    }
  ]);  
  
  //Creating the complete Tree of Thoughts sequence
  const treeOfThoughtsChain = RunnableSequence.from([
    //Preparing the initial data
    (input) => {
      console.log("\n🌳 STARTING TREE OF THOUGHTS PROCESS 🌳");
      console.log(`Problem to solve: ${input.input}`);
      return { ...input, original_input: input.input };
    },
    //Executing Step 1: Brainstorming solutions
    step1Chain,
    //Executing Step 2: Evaluation of pros and cons
    step2Chain,
    //Executing Step 3: Final recommendation
    step3Chain,
    //Executing Step 4: Final refinement
    step4Chain,
    //Formatting the final output
    (output) => {
      console.log("\n🏆 COMPLETE PROCESS FINISHED 🏆");
      console.log("Final recommendation generated successfully!");
      return {
        final_recommendation: output.final_recommendation,
        processo_completo: output.raw_output
      };
    }
  ]);

//Simple prompt, that tries to solve the problem in a more direct way, without using the Tree of Thoughts.
//Implementation of the prompt template to solve the problem in a more direct way.
const simplePromptTemplate = new PromptTemplate({
  template: `
  You are a helpful assistant that can solve problems.
  The problem is: {problem}
  Please provide a solution to the problem.
  `,
  inputVariables: ["problem"]
});

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",
  temperature: 1.0,
});

const chain = simplePromptTemplate.pipe(llm);


async function main() {
  //First solve the problem with the simple prompt template
  const response1 = await chain.invoke({
    problem: "Human colonization of Mars, considering that the distance between Earth and Mars is very large, making regular resupply difficult"
  });
  console.log(`😐 PROBLEM SOLVED WITH SIMPLE PROMPT TEMPLATE 😐:\n${response1.content}\n\n\n`);

  //Now solve the problem with the Tree of Thoughts
  const response2 = await treeOfThoughtsChain.invoke({
    input: "human colonization of Mars",
    perfect_factors: "The distance between Earth and Mars is very large, making regular resupply difficult"
  });
  
  console.log(response2);
}

main().catch(console.error);
