from pluggy import HookimplMarker
from pluggy import HookspecMarker

hookspec = HookspecMarker("llm")
hookimpl = HookimplMarker("llm")


@hookspec
def register_commands(cli):
    """Register additional CLI commands, e.g. 'llm mycommand ...'"""


@hookspec
def register_models(register):
    "Register additional model instances representing LLM models that can be called"


@hookspec
def register_embedding_models(register):
    "Register additional model instances that can be used for embedding"


@hookspec
def register_template_loaders(register):
    "Register additional template loaders with prefixes"


@hookspec
def register_fragment_loaders(register):
    "Register additional fragment loaders with prefixes"


@hookspec
def register_tools(register):
    "Register functions that can be used as tools by the LLMs"


@hookspec
def register_prompt_executor(register):
    "Register a custom PromptExecutor class for handling prompt execution UI"


@hookspec
def register_chat_interface(register):
    "Register a custom ChatInterface class for handling chat UI"
