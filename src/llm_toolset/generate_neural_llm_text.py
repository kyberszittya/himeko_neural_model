import sys

from openai import OpenAI


class GenerateNeuralNetworkCode:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    # Function to generate code based on meta-element and description
    def generate_code(self, meta_description: str, description: str, framework="python"):
        """
        Generate code based on a meta-element description and user description.

        Parameters:
        - meta_description (str): High-level description of the function or module.
        - description (str): Additional details or specific requirements.
        - framework (str): The language or framework for code generation (e.g., 'pytorch', 'tensorflow', 'python').

        Returns:
        - str: Generated code from OpenAI API.
        """

        # Construct prompt with meta-description and user description
        prompt = (f"Generate {framework} code based from the given description that is based on the given metamodel:\n\n"
                  f"Meta-element: {meta_description}\n"
                  f"Description: {description}\n\n"
                  f"Code:\n")

        try:
            # Call OpenAI API with the prompt
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a code generator, use provided files to generate code. Give only code back (don't include any other message), and make a runnable code printing the architecture. remove markdown indicatives (i.e. ```python) and comments."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract code from the response
            generated_code = response.choices[0].message.content.strip()
            return generated_code

        except Exception as e:
            return f"An error occurred: {str(e)}"


def main(args):
    # Example usage
    framework = "pytorch"
    meta_description_file = args[1]
    meta_description = open(meta_description_file).readlines()
    description_file = args[2]
    description = open(description_file).readlines()
    print(meta_description)
    print(description)
    ai_key = open("openai_api.key").read().strip()
    code_generator = GenerateNeuralNetworkCode(ai_key)
    for i in range(1):
        generated_code = code_generator.generate_code(meta_description, description, framework=framework)
        print(f"Generated Code {i}:\n", generated_code)
        with open(f"./testcodes/generated_code_{i}.py", "w") as f:
            f.write(generated_code)
        print(f"Generated Code {i} -- STOP")


if __name__=="__main__":
    main(sys.argv)
