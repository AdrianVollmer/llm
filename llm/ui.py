"""
User interface components for LLM CLI.

This module contains the UI logic for prompt execution and chat interfaces,
separated from the CLI argument parsing in cli.py.

The main classes are:
- PromptExecutor: Base class for prompt execution UI (can be subclassed by plugins)
- ChatInterface: Base class for chat REPL UI (can be subclassed by plugins)

Plugins can extend these classes to customize the UI behavior.
"""

import asyncio
import click
import os
import sys
from typing import Optional, Callable, Union

from llm import (
    Attachment,
    AsyncResponse,
    Fragment,
    Template,
)
from llm.models import ChainResponse


class PromptExecutor:
    """
    Base class for prompt execution UI.

    This class handles the execution of a prompt and displaying the response.
    Plugins can subclass this to customize the UI behavior.

    Extension points (methods that can be overridden):
    - read_input(): Read the prompt input
    - resolve_fragments(): Resolve fragment and attachment references
    - prepare_execution(): Set up the prompt method and kwargs
    - execute(): Execute the prompt (sync or async)
    - handle_streaming(): Handle streaming output
    - handle_async_response(): Handle async responses
    - format_output(): Format text output (e.g., extract code blocks)
    - print_output(): Print output to stdout
    - handle_error(): Handle exceptions
    - display_usage(): Display token usage information
    - log_response(): Log response to database
    """

    def __init__(
        self,
        prompt: str,
        db,
        model,
        conversation,
        system: Optional[str],
        schema,
        fragments: list,
        system_fragments: list,
        resolved_attachments: list,
        should_stream: bool,
        async_: bool,
        extract: bool,
        extract_last: bool,
        tool_implementations,
        tools_debug: bool,
        tools_approve: bool,
        chain_limit: int,
        validated_options: dict,
        usage: bool,
        no_log: bool,
        log: bool,
        resolve_fragments_fn: Callable,
        extract_fenced_code_block_fn: Callable,
        logs_on_fn: Callable,
        _debug_tool_call_fn: Optional[Callable],
        _approve_tool_call_fn: Optional[Callable],
    ):
        """Initialize the PromptExecutor with all necessary parameters."""
        self.prompt = prompt
        self.db = db
        self.model = model
        self.conversation = conversation
        self.system = system
        self.schema = schema
        self.fragments = fragments
        self.system_fragments = system_fragments
        self.resolved_attachments = resolved_attachments
        self.should_stream = should_stream
        self.async_ = async_
        self.extract = extract
        self.extract_last = extract_last
        self.tool_implementations = tool_implementations
        self.tools_debug = tools_debug
        self.tools_approve = tools_approve
        self.chain_limit = chain_limit
        self.validated_options = validated_options
        self.usage = usage
        self.no_log = no_log
        self.log = log
        self.resolve_fragments_fn = resolve_fragments_fn
        self.extract_fenced_code_block_fn = extract_fenced_code_block_fn
        self.logs_on_fn = logs_on_fn
        self._debug_tool_call_fn = _debug_tool_call_fn
        self._approve_tool_call_fn = _approve_tool_call_fn
        self.response = None

    def resolve_fragments(self):
        """Resolve fragment and attachment references."""
        from llm.errors import FragmentNotFound

        try:
            fragments_and_attachments = self.resolve_fragments_fn(self.db, self.fragments, allow_attachments=True)
            resolved_fragments = [fragment for fragment in fragments_and_attachments if isinstance(fragment, Fragment)]
            self.resolved_attachments.extend(attachment for attachment in fragments_and_attachments if isinstance(attachment, Attachment))
            resolved_system_fragments = self.resolve_fragments_fn(self.db, self.system_fragments)
            return resolved_fragments, resolved_system_fragments
        except FragmentNotFound as ex:
            raise click.ClickException(str(ex))

    def prepare_execution(self, resolved_fragments, resolved_system_fragments):
        """Prepare the prompt method and kwargs for execution."""
        prompt_method = self.model.prompt
        if self.conversation:
            prompt_method = self.conversation.prompt

        kwargs = {}

        if self.tool_implementations:
            prompt_method = self.conversation.chain
            kwargs["options"] = self.validated_options
            kwargs["chain_limit"] = self.chain_limit
            if self.tools_debug:
                kwargs["after_call"] = self._debug_tool_call_fn
            if self.tools_approve:
                kwargs["before_call"] = self._approve_tool_call_fn
            kwargs["tools"] = self.tool_implementations
        else:
            kwargs.update(self.validated_options)

        kwargs.update(
            {
                "attachments": self.resolved_attachments,
                "system": self.system,
                "schema": self.schema,
                "fragments": resolved_fragments,
                "system_fragments": resolved_system_fragments,
            }
        )

        return prompt_method, kwargs

    def handle_streaming(self, response):
        """Handle streaming output."""
        for chunk in response:
            print(chunk, end="")
            sys.stdout.flush()
        print("")

    async def handle_async_streaming(self, response):
        """Handle async streaming output."""
        async for chunk in response:
            print(chunk, end="")
            sys.stdout.flush()
        print("")

    def format_output(self, text: str) -> str:
        """Format text output (e.g., extract code blocks)."""
        if self.extract or self.extract_last:
            return self.extract_fenced_code_block_fn(text, last=self.extract_last) or text
        return text

    def print_output(self, text: str):
        """Print output to stdout."""
        print(text)

    async def execute_async(self, prompt_method, kwargs):
        """Execute the prompt asynchronously."""
        if self.should_stream:
            response = prompt_method(self.prompt, **kwargs)
            await self.handle_async_streaming(response)
        else:
            response = prompt_method(self.prompt, **kwargs)
            text = await response.text()
            text = self.format_output(text)
            self.print_output(text)
        return response

    def execute_sync(self, prompt_method, kwargs):
        """Execute the prompt synchronously."""
        response = prompt_method(self.prompt, **kwargs)
        if self.should_stream:
            self.handle_streaming(response)
        else:
            text = response.text()
            text = self.format_output(text)
            self.print_output(text)
        return response

    def handle_error(self, ex: Exception):
        """Handle exceptions during execution."""
        if isinstance(ex, (ValueError, NotImplementedError)):
            raise click.ClickException(str(ex))
        if getattr(sys, "_called_from_test", False) or os.environ.get("LLM_RAISE_ERRORS", None):
            raise
        raise click.ClickException(str(ex))

    def display_usage(self):
        """Display token usage information."""
        if self.usage and self.response:
            if isinstance(self.response, ChainResponse):
                responses = self.response._responses
            else:
                responses = [self.response]
            for response_object in responses:
                click.echo(
                    click.style(
                        "Token usage: {}".format(response_object.token_usage()),
                        fg="yellow",
                        bold=True,
                    ),
                    err=True,
                )

    def log_response(self):
        """Log response to database."""
        if (self.logs_on_fn() or self.log) and not self.no_log:
            if isinstance(self.response, AsyncResponse):
                self.response = asyncio.run(self.response.to_sync_response())
            self.response.log_to_db(self.db)

    def execute(self):
        """
        Execute the prompt and return the response.

        This is the main entry point that orchestrates the entire execution flow.
        """
        resolved_fragments, resolved_system_fragments = self.resolve_fragments()
        prompt_method, kwargs = self.prepare_execution(resolved_fragments, resolved_system_fragments)

        try:
            if self.async_:
                self.response = asyncio.run(self.execute_async(prompt_method, kwargs))
            else:
                self.response = self.execute_sync(prompt_method, kwargs)
        except Exception as ex:
            self.handle_error(ex)

        self.display_usage()
        self.log_response()

        return self.response


def execute_prompt_ui(
    prompt: str,
    db,
    model,
    conversation,
    system: Optional[str],
    schema,
    fragments,
    system_fragments,
    resolved_attachments: list,
    should_stream: bool,
    async_: bool,
    extract: bool,
    extract_last: bool,
    tool_implementations,
    tools_debug,
    tools_approve,
    chain_limit: int,
    validated_options: dict,
    usage: bool,
    no_log: bool,
    log: bool,
    resolve_fragments_fn,
    extract_fenced_code_block_fn,
    logs_on_fn,
    _debug_tool_call_fn,
    _approve_tool_call_fn,
):
    """
    Execute a prompt and handle the UI for displaying the response.

    This is a functional wrapper around PromptExecutor for backward compatibility.

    Args:
        prompt: The user's prompt text
        db: Database connection
        model: The LLM model to use
        conversation: Optional conversation context
        system: System prompt
        schema: JSON schema for structured output
        fragments: List of fragment identifiers
        system_fragments: List of system fragment identifiers
        resolved_attachments: Pre-resolved attachments list
        should_stream: Whether to stream the response
        async_: Whether to use async execution
        extract: Whether to extract fenced code blocks
        extract_last: Whether to extract only the last code block
        tool_implementations: Tools available to the model
        tools_debug: Whether to show debug info for tools
        tools_approve: Whether to require approval for tool calls
        chain_limit: Maximum number of tool chain iterations
        validated_options: Model options
        usage: Whether to display token usage
        no_log: Whether to skip logging
        log: Whether to force logging
        resolve_fragments_fn: Function to resolve fragments
        extract_fenced_code_block_fn: Function to extract code blocks
        logs_on_fn: Function to check if logging is enabled
        _debug_tool_call_fn: Callback for tool debugging
        _approve_tool_call_fn: Callback for tool approval

    Returns:
        The response object

    Raises:
        click.ClickException: For user-facing errors
    """
    # Get the PromptExecutor class (may be overridden by plugins)
    # If stdout is piped, bypass plugins and use the default executor
    executor_class = get_prompt_executor(bypass_plugins=not sys.stdout.isatty())

    executor = executor_class(
        prompt=prompt,
        db=db,
        model=model,
        conversation=conversation,
        system=system,
        schema=schema,
        fragments=fragments,
        system_fragments=system_fragments,
        resolved_attachments=resolved_attachments,
        should_stream=should_stream,
        async_=async_,
        extract=extract,
        extract_last=extract_last,
        tool_implementations=tool_implementations,
        tools_debug=tools_debug,
        tools_approve=tools_approve,
        chain_limit=chain_limit,
        validated_options=validated_options,
        usage=usage,
        no_log=no_log,
        log=log,
        resolve_fragments_fn=resolve_fragments_fn,
        extract_fenced_code_block_fn=extract_fenced_code_block_fn,
        logs_on_fn=logs_on_fn,
        _debug_tool_call_fn=_debug_tool_call_fn,
        _approve_tool_call_fn=_approve_tool_call_fn,
    )
    return executor.execute()


class ChatInterface:
    """
    Base class for chat REPL UI.

    This class handles the interactive chat interface with a model.
    Plugins can subclass this to customize the UI behavior.

    Extension points (methods that can be overridden):
    - print_welcome(): Display welcome messages and instructions
    - run(): Main REPL loop
    - read_prompt(): Read a single prompt from the user
    - read_multiline_prompt(): Read multi-line input with advanced editing
    - handle_special_command(): Process special commands (!multi, !edit, etc.)
    - handle_multi_line(): Handle multi-line input mode
    - handle_edit_command(): Open editor for prompt editing
    - handle_fragment_command(): Process fragment references
    - should_exit(): Check if the user wants to exit
    - prepare_prompt(): Prepare the prompt with template processing
    - execute_prompt(): Execute a single chat turn
    - display_response(): Display the response with streaming
    """

    def __init__(
        self,
        db,
        model,
        conversation,
        system: Optional[str],
        argument_fragments: list,
        argument_attachments: list,
        argument_system_fragments: list,
        template_obj: Optional[Template],
        params: dict,
        kwargs: dict,
        process_fragments_in_chat_fn: Callable,
    ):
        """Initialize the ChatInterface with all necessary parameters."""
        self.db = db
        self.model = model
        self.conversation = conversation
        self.system = system
        self.argument_fragments = argument_fragments
        self.argument_attachments = argument_attachments
        self.argument_system_fragments = argument_system_fragments
        self.template_obj = template_obj
        self.params = params
        self.kwargs = kwargs
        self.process_fragments_in_chat_fn = process_fragments_in_chat_fn

        # State for multi-line mode
        self.in_multi = False
        self.accumulated = []
        self.accumulated_fragments = []
        self.accumulated_attachments = []
        self.end_token = "!end"

    def print_welcome(self):
        """Display welcome messages and instructions."""
        click.echo("Chatting with {}".format(self.model.model_id))
        click.echo("Type 'exit' or 'quit' to exit")
        click.echo("Type '!multi' to enter multiple lines, then '!end' to finish")
        click.echo("Type '!edit' to open your default editor and modify the prompt")
        click.echo("Type '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments")

    def read_prompt(self) -> str:
        """
        Read a single prompt from the user.

        This is called for single-line input or for each line in basic multi-line mode.
        Plugins can override this to provide custom input handling (e.g., with history).
        """
        return click.prompt("", prompt_suffix="> " if not self.in_multi else "")

    def read_multiline_prompt(self) -> Optional[str]:
        """
        Read a multi-line prompt from the user with advanced editing capabilities.

        This is called when the user enters multi-line mode with !multi.
        The default implementation returns None, which falls back to line-by-line input.

        Plugins can override this to provide advanced multi-line editing with features like:
        - Cursor movement across lines
        - Command history
        - Syntax highlighting

        Returns:
            The complete multi-line prompt text, or None to use default line-by-line mode.
        """
        return None

    def should_exit(self, prompt: str) -> bool:
        """Check if the user wants to exit."""
        return prompt.strip() in ("exit", "quit")

    def handle_multi_line(self, prompt: str) -> Union[tuple, bool, None]:
        """
        Handle multi-line input mode.

        Returns:
            - True if we should continue to next iteration (accumulating lines)
            - False if not in multi-line mode
            - tuple of (prompt, fragments, attachments) if advanced multi-line mode completed
        """
        if prompt.strip().startswith("!multi"):
            # Try advanced multi-line editor first
            multiline_text = self.read_multiline_prompt()
            if multiline_text is not None:
                # Advanced editor provided the full text
                return (multiline_text, [], [])
            else:
                # Fall back to line-by-line mode
                self.in_multi = True
                bits = prompt.strip().split()
                if len(bits) > 1:
                    self.end_token = "!end {}".format(" ".join(bits[1:]))
                return True
        return False

    def handle_edit_command(self, prompt: str) -> Optional[str]:
        """
        Handle the !edit command to open an editor.

        Returns the edited prompt or None if editing was cancelled.
        """
        if prompt.strip() == "!edit":
            edited_prompt = click.edit()
            if edited_prompt is None:
                click.echo("Editor closed without saving.", err=True)
                return None
            return edited_prompt.strip()
        return prompt

    def handle_fragment_command(self, prompt: str):
        """
        Handle the !fragment command.

        Returns tuple of (prompt, fragments, attachments).
        """
        if prompt.strip().startswith("!fragment "):
            return self.process_fragments_in_chat_fn(self.db, prompt)
        return prompt, [], []

    def accumulate_input(self, prompt: str, fragments: list, attachments: list) -> bool:
        """
        Accumulate input in multi-line mode.

        Returns True if we should continue to next iteration, False if ready to execute.
        """
        if self.in_multi:
            if prompt.strip() == self.end_token:
                # Multi-line input complete
                return False
            else:
                # Still accumulating
                if prompt:
                    self.accumulated.append(prompt)
                self.accumulated_fragments += fragments
                self.accumulated_attachments += attachments
                return True
        return False

    def finalize_multiline(self):
        """
        Finalize multi-line input and return the accumulated data.

        Returns tuple of (prompt, fragments, attachments).
        """
        prompt = "\n".join(self.accumulated)
        fragments = self.accumulated_fragments
        attachments = self.accumulated_attachments

        # Reset state
        self.in_multi = False
        self.accumulated = []
        self.accumulated_fragments = []
        self.accumulated_attachments = []
        self.end_token = "!end"

        return prompt, fragments, attachments

    def prepare_prompt(self, prompt: str) -> str:
        """Apply template processing to the prompt if needed."""
        if self.template_obj:
            try:
                uses_input = "input" in self.template_obj.vars()
                input_ = prompt if uses_input else ""
                template_prompt, template_system = self.template_obj.evaluate(input_, self.params)
            except Template.MissingVariables as ex:
                raise click.ClickException(str(ex))

            if template_system and not self.system:
                self.system = template_system

            if template_prompt:
                if prompt and not uses_input:
                    prompt = f"{template_prompt}\n{prompt}"
                else:
                    prompt = template_prompt

        return prompt

    def execute_prompt(self, prompt: str, fragments: list, attachments: list):
        """Execute a single chat turn and return the response."""
        response = self.conversation.chain(
            prompt,
            fragments=[str(fragment) for fragment in fragments],
            system_fragments=[str(system_fragment) for system_fragment in self.argument_system_fragments],
            attachments=attachments,
            system=self.system,
            **self.kwargs,
        )

        # System prompt and system fragments only sent for the first message
        self.system = None
        self.argument_system_fragments = []

        return response

    def display_response(self, response):
        """Display the response with streaming."""
        for chunk in response:
            print(chunk, end="")
            sys.stdout.flush()
        response.log_to_db(self.db)
        print("")

    def run(self):
        """Main REPL loop."""
        self.print_welcome()

        while True:
            prompt = self.read_prompt()
            fragments = []
            attachments = []

            # Add argument fragments/attachments only to first message
            if self.argument_fragments:
                fragments += self.argument_fragments
                self.argument_fragments = []
            if self.argument_attachments:
                attachments = self.argument_attachments
                self.argument_attachments = []

            # Handle !multi command
            multi_result = self.handle_multi_line(prompt)
            if multi_result is True:
                # Continue accumulating lines in basic mode
                continue
            elif isinstance(multi_result, tuple):
                # Advanced multi-line editor provided complete input
                prompt, fragments, attachments = multi_result
                # Skip to execution (bypass other handlers for this prompt)
                prompt = self.prepare_prompt(prompt)
                if self.should_exit(prompt):
                    break
                response = self.execute_prompt(prompt, fragments, attachments)
                self.display_response(response)
                continue

            # Handle !edit command
            edited = self.handle_edit_command(prompt)
            if edited is None:
                continue
            prompt = edited

            # Handle !fragment command
            prompt, frag_list, attach_list = self.handle_fragment_command(prompt)
            fragments += frag_list
            attachments += attach_list

            # Handle multi-line accumulation
            if self.accumulate_input(prompt, fragments, attachments):
                continue

            # Finalize multi-line if needed
            if self.in_multi is False and self.accumulated:
                prompt, fragments, attachments = self.finalize_multiline()

            # Apply template processing
            prompt = self.prepare_prompt(prompt)

            # Check for exit
            if self.should_exit(prompt):
                break

            # Execute and display
            response = self.execute_prompt(prompt, fragments, attachments)
            self.display_response(response)


def execute_chat_ui(
    db,
    model,
    conversation,
    system: Optional[str],
    argument_fragments: list,
    argument_attachments: list,
    argument_system_fragments: list,
    template_obj: Optional[Template],
    params: dict,
    kwargs: dict,
    process_fragments_in_chat_fn,
):
    """
    Execute a chat REPL interface.

    This is a functional wrapper around ChatInterface for backward compatibility.

    Args:
        db: Database connection
        model: The LLM model to use
        conversation: Conversation context
        system: System prompt (for first message only)
        argument_fragments: Initial fragments from command line
        argument_attachments: Initial attachments from command line
        argument_system_fragments: System fragments from command line
        template_obj: Optional template for processing prompts
        params: Template parameters
        kwargs: Additional keyword arguments for conversation.chain()
        process_fragments_in_chat_fn: Function to process fragment commands
    """
    # Get the ChatInterface class (may be overridden by plugins)
    interface_class = get_chat_interface()

    interface = interface_class(
        db=db,
        model=model,
        conversation=conversation,
        system=system,
        argument_fragments=argument_fragments,
        argument_attachments=argument_attachments,
        argument_system_fragments=argument_system_fragments,
        template_obj=template_obj,
        params=params,
        kwargs=kwargs,
        process_fragments_in_chat_fn=process_fragments_in_chat_fn,
    )
    interface.run()


def get_prompt_executor(bypass_plugins: bool = False) -> type:
    """
    Get the PromptExecutor class to use for prompt execution UI.

    Plugins can register custom PromptExecutor classes via the
    register_prompt_executor hook. If no plugin is registered,
    returns the default PromptExecutor class.

    Args:
        bypass_plugins: If True, ignore plugin executors and return the default.
                       This is used when stdout is piped to ensure plain text output.

    Returns:
        The PromptExecutor class to instantiate
    """
    # If bypassing plugins (e.g., when stdout is piped), return default immediately
    if bypass_plugins:
        return PromptExecutor

    from .plugins import pm

    # Collect all registered prompt executors from plugins
    custom_executors = []

    def _register(executor_class):
        custom_executors.append(executor_class)

    pm.hook.register_prompt_executor(register=_register)

    # If a plugin registered an executor, use the last one
    # (allows plugins to override each other with predictable behavior)
    if custom_executors:
        return custom_executors[-1]

    # Otherwise return the default
    return PromptExecutor


def get_chat_interface() -> type:
    """
    Get the ChatInterface class to use for chat UI.

    Plugins can register custom ChatInterface classes via the
    register_chat_interface hook. If no plugin is registered,
    returns the default ChatInterface class.

    Returns:
        The ChatInterface class to instantiate
    """
    from .plugins import pm

    # Collect all registered chat interfaces from plugins
    custom_interfaces = []

    def _register(interface_class):
        custom_interfaces.append(interface_class)

    pm.hook.register_chat_interface(register=_register)

    # If a plugin registered an interface, use the last one
    # (allows plugins to override each other with predictable behavior)
    if custom_interfaces:
        return custom_interfaces[-1]

    # Otherwise return the default
    return ChatInterface
