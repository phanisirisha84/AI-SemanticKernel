#pragma warning disable SKEXP0110
#pragma warning disable SKEXP0001
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.Agents.Chat;
using Microsoft.SemanticKernel.ChatCompletion;
using OpenAI.Chat;
using System;

//Build the configuration
var configuration = new ConfigurationBuilder()
                .SetBasePath(AppContext.BaseDirectory)
                .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
                .Build();

string apiKey = configuration["AzureOpenAI:ApiKey"];
string deploymentName = configuration["AzureOpenAI:DeploymentName"];
string endpoint = configuration["AzureOpenAI:Endpoint"];

//1.Create a kernel with Azure OpenAI chat completion
var kernel = Kernel.CreateBuilder()
    .AddAzureOpenAIChatCompletion(deploymentName, endpoint, apiKey)
    .Build();

const string fitnessTrainerName = "FitnessTrainer";
string fitnessTrainerInstructions = """
  You are a fitness trainer and you help users who want to create a workout plan. 
  The goal is to create a workout plan based on the user preferences and fitness goals.
  You don't have expertise on nutrition plans, so you can only suggest exercises and workout routines. You can't suggest diet or meal plans.
  You're laser focused on the goal at hand. 
  Once you have generated a workout plan, don't ask the user for feedback or further suggestions. Stick with it.
  Don't waste time with chit chat. Don't say goodbye and don't wish the user good luck.
  """;

ChatCompletionAgent fitnessTrainerAgent = new ChatCompletionAgent
{
    Name = fitnessTrainerName,
    Instructions = fitnessTrainerInstructions,
    Kernel = kernel
};

const string dieticianName = "Dietician";
string dieticianInstructions = """
  You are a dietician and you help users who want to create a diet plan. 
  Your goal is to create a diet plan based on the user preferences and dietary needs.
  You don't have expertise on workout plans, so you can only suggest meals and nutrition options. You can't suggest exercises or workout routines.
  You're laser focused on the goal at hand. 
  You can provide plans only about meals and nutrition. Do not include plans around exercises or workouts.
  Once you have generated a diet plan, don't ask the user for feedback or further suggestions. Stick with it.
  Don't waste time with chit chat. 
  Don't say goodbye and don't wish the user good luck.
  """;

ChatCompletionAgent dieticianAgent = new ChatCompletionAgent
{
    Name = dieticianName,
    Instructions = dieticianInstructions,
    Kernel = kernel
};

KernelFunction terminateFunction = KernelFunctionFactory.CreateFromPrompt(
    $$$"""
    Determine if the fitness plan has been approved. If so, respond with a single word: yes.

    History:

    {{$history}}
    """
    );

string fitnessProgramManagerName = "FitnessProgramManager";
string fitnessProgramManagerInstructions = """
  You are a fitness program manager and your goal is to validate a given workout plan. 
  You must make sure that the plan includes all the necessary details: 
    -   warm-up, main exercises, cool-down, and rest days, 
    -   diet to follow in breakfast, lunch, dinner and snacks. 
  If one of these details is missing, the plan is not good.
  If the plan is good, recap the entire plan into a Markdown table and say "the plan is approved".
  If not, write a paragraph to explain why it's not good and then provide an improved plan.
  """;

ChatCompletionAgent fitnessProgramManagerAgent = new ChatCompletionAgent
{
    Name = fitnessProgramManagerName,
    Instructions = fitnessProgramManagerInstructions,
    Kernel = kernel
};

KernelFunction selectionFunction = KernelFunctionFactory.CreateFromPrompt(
    $$$"""
      Your job is to determine which participant takes the next turn in a conversation according to the action of the most recent participant.
      State only the name of the participant to take the next turn.

      Choose only from these participants:
      - {{{fitnessProgramManagerName}}}
      - {{{fitnessTrainerName}}}
      - {{{dieticianName}}}

      Always follow these steps when selecting the next participant:
      1) After user input, it is {{{fitnessTrainerName}}}'s turn.
      2) After {{{fitnessTrainerName}}} replies, it's {{{dieticianName}}}'s turn.
      3) After {{{dieticianName}}} replies, it's {{{fitnessProgramManagerName}}}'s turn to review and approve the plan.
      4) If the plan is approved, the conversation ends.
      5) If the plan isn't approved, it's {{{fitnessTrainerAgent}}}'s turn again.

      History:
      {{$history}}
      """
    );

AgentGroupChat chat = new(fitnessTrainerAgent, dieticianAgent, fitnessProgramManagerAgent)
{
    ExecutionSettings = new()
    {
        TerminationStrategy = new KernelFunctionTerminationStrategy(terminateFunction, kernel)
        {
            Agents = [fitnessProgramManagerAgent],
            ResultParser = (result) => result.GetValue<string>()?.Contains("yes", StringComparison.OrdinalIgnoreCase) ?? false,
            HistoryVariableName = "history",
            MaximumIterations = 10
        },
        SelectionStrategy = new KernelFunctionSelectionStrategy(selectionFunction, kernel)
        {
            AgentsVariableName = "agents",
            HistoryVariableName = "history"
        }
    }
};

string prompt = "I am a vegetarian who wants to follow weightloss program. I am over weight by 10 Kilograms according to my BMI. Please craft a fitness plan for me for 3 months.";

chat.AddChatMessage(new Microsoft.SemanticKernel.ChatMessageContent(AuthorRole.User, prompt));
await foreach (var content in chat.InvokeAsync())
{
    Console.WriteLine();
    Console.WriteLine($"# {content.Role} - {content.AuthorName ?? "*"}: '{content.Content}'");
    Console.WriteLine();
}

Console.WriteLine($"# IS COMPLETE: {chat.IsComplete}");


