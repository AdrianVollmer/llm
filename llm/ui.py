"""
User interface components for LLM CLI.

This module contains the UI logic for prompt execution and chat interfaces,
separated from the CLI argument parsing in cli.py.
"""

import asyncio
import click
import os
import sys
from typing import Optional

from llm import (
    Attachment,
    AsyncResponse,
    Fragment,
    Template,
)
from llm.models import ChainResponse


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

    This function handles:
    - Fragment and attachment resolution
    - Response execution (sync/async, streaming/non-streaming)
    - Output formatting and printing
    - Error handling
    - Token usage display
    - Logging to database

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
    from llm.errors import FragmentNotFound

    response = None

    try:
        fragments_and_attachments = resolve_fragments_fn(
            db, fragments, allow_attachments=True
        )
        resolved_fragments = [
            fragment
            for fragment in fragments_and_attachments
            if isinstance(fragment, Fragment)
        ]
        resolved_attachments.extend(
            attachment
            for attachment in fragments_and_attachments
            if isinstance(attachment, Attachment)
        )
        resolved_system_fragments = resolve_fragments_fn(db, system_fragments)
    except FragmentNotFound as ex:
        raise click.ClickException(str(ex))

    prompt_method = model.prompt
    if conversation:
        prompt_method = conversation.prompt

    kwargs = {}

    if tool_implementations:
        prompt_method = conversation.chain
        kwargs["options"] = validated_options
        kwargs["chain_limit"] = chain_limit
        if tools_debug:
            kwargs["after_call"] = _debug_tool_call_fn
        if tools_approve:
            kwargs["before_call"] = _approve_tool_call_fn
        kwargs["tools"] = tool_implementations
    else:
        # Merge in options for the .prompt() methods
        kwargs.update(validated_options)

    try:
        if async_:

            async def inner():
                if should_stream:
                    response = prompt_method(
                        prompt,
                        attachments=resolved_attachments,
                        system=system,
                        schema=schema,
                        fragments=resolved_fragments,
                        system_fragments=resolved_system_fragments,
                        **kwargs,
                    )
                    async for chunk in response:
                        print(chunk, end="")
                        sys.stdout.flush()
                    print("")
                else:
                    response = prompt_method(
                        prompt,
                        fragments=resolved_fragments,
                        attachments=resolved_attachments,
                        schema=schema,
                        system=system,
                        system_fragments=resolved_system_fragments,
                        **kwargs,
                    )
                    text = await response.text()
                    if extract or extract_last:
                        text = (
                            extract_fenced_code_block_fn(text, last=extract_last) or text
                        )
                    print(text)
                return response

            response = asyncio.run(inner())
        else:
            response = prompt_method(
                prompt,
                fragments=resolved_fragments,
                attachments=resolved_attachments,
                system=system,
                schema=schema,
                system_fragments=resolved_system_fragments,
                **kwargs,
            )
            if should_stream:
                for chunk in response:
                    print(chunk, end="")
                    sys.stdout.flush()
                print("")
            else:
                text = response.text()
                if extract or extract_last:
                    text = extract_fenced_code_block_fn(text, last=extract_last) or text
                print(text)
    except (ValueError, NotImplementedError) as ex:
        raise click.ClickException(str(ex))
    except Exception as ex:
        if getattr(sys, "_called_from_test", False) or os.environ.get(
            "LLM_RAISE_ERRORS", None
        ):
            raise
        raise click.ClickException(str(ex))

    if usage:
        if isinstance(response, ChainResponse):
            responses = response._responses
        else:
            responses = [response]
        for response_object in responses:
            click.echo(
                click.style(
                    "Token usage: {}".format(response_object.token_usage()),
                    fg="yellow",
                    bold=True,
                ),
                err=True,
            )

    if (logs_on_fn() or log) and not no_log:
        if isinstance(response, AsyncResponse):
            response = asyncio.run(response.to_sync_response())
        response.log_to_db(db)

    return response


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

    This function handles:
    - Printing welcome messages and help text
    - Reading user input in a loop
    - Processing special commands (!multi, !edit, !fragment, exit/quit)
    - Multi-line input accumulation
    - Executing chat turns with the model
    - Displaying streaming responses
    - Logging responses to database

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
    click.echo("Chatting with {}".format(model.model_id))
    click.echo("Type 'exit' or 'quit' to exit")
    click.echo("Type '!multi' to enter multiple lines, then '!end' to finish")
    click.echo("Type '!edit' to open your default editor and modify the prompt")
    click.echo(
        "Type '!fragment <my_fragment> [<another_fragment> ...]' to insert one or more fragments"
    )
    in_multi = False

    accumulated = []
    accumulated_fragments = []
    accumulated_attachments = []
    end_token = "!end"
    while True:
        prompt = click.prompt("", prompt_suffix="> " if not in_multi else "")
        fragments = []
        attachments = []
        if argument_fragments:
            fragments += argument_fragments
            argument_fragments = []
        if argument_attachments:
            attachments = argument_attachments
            argument_attachments = []
        if prompt.strip().startswith("!multi"):
            in_multi = True
            bits = prompt.strip().split()
            if len(bits) > 1:
                end_token = "!end {}".format(" ".join(bits[1:]))
            continue
        if prompt.strip() == "!edit":
            edited_prompt = click.edit()
            if edited_prompt is None:
                click.echo("Editor closed without saving.", err=True)
                continue
            prompt = edited_prompt.strip()
        if prompt.strip().startswith("!fragment "):
            prompt, fragments, attachments = process_fragments_in_chat_fn(db, prompt)

        if in_multi:
            if prompt.strip() == end_token:
                prompt = "\n".join(accumulated)
                fragments = accumulated_fragments
                attachments = accumulated_attachments
                in_multi = False
                accumulated = []
                accumulated_fragments = []
                accumulated_attachments = []
            else:
                if prompt:
                    accumulated.append(prompt)
                accumulated_fragments += fragments
                accumulated_attachments += attachments
                continue
        if template_obj:
            try:
                uses_input = "input" in template_obj.vars()
                input_ = prompt if uses_input else ""
                template_prompt, template_system = template_obj.evaluate(input_, params)
            except Template.MissingVariables as ex:
                raise click.ClickException(str(ex))
            if template_system and not system:
                system = template_system
            if template_prompt:
                if prompt and not uses_input:
                    prompt = f"{template_prompt}\n{prompt}"
                else:
                    prompt = template_prompt
        if prompt.strip() in ("exit", "quit"):
            break

        response = conversation.chain(
            prompt,
            fragments=[str(fragment) for fragment in fragments],
            system_fragments=[
                str(system_fragment) for system_fragment in argument_system_fragments
            ],
            attachments=attachments,
            system=system,
            **kwargs,
        )

        system = None
        argument_system_fragments = []
        for chunk in response:
            print(chunk, end="")
            sys.stdout.flush()
        response.log_to_db(db)
        print("")
