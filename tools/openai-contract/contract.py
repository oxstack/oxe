"""OpenAI v1 contract suite: the real openai-python SDK against the local server (v0.2.0 M3).

Run by run.sh (next to this file) in one of two modes; nothing here starts a server:
  OAC_MODE=engine      OAC_URL / OAC_TOOLS_URL = `oxe-server --engine stub` itself
  OAC_MODE=controller  OAC_URL / OAC_TOOLS_URL = the lab controller (queue ON) in front of it;
                       OAC_IDLE_URL = a second controller that must never load its model
OAC_ENGINE_URL: the engine behind OAC_URL (controller mode: paired direct-vs-controller checks).
OAC_TOOLS_URL serves a stub whose reply is two tool calls (get_weather, get_time).
OAC_SLOW=1: the engine prefills slowly (`--stub-prefill-us`), so the queue and replay tests
can cut a request in flight. OAC_DEFAULT_MODEL: a model whose output default is 4 tokens
(controller mode: the profile's max_output_tokens).

Generic only (O-133): no client name anywhere; the SDK is the client. Each test's modes are
pinned in tests.txt; a test outside the current mode skips itself and run_contract.py checks that
exactly the pinned set ran.
"""
import json
import os
import time
import unittest
import urllib.request
import uuid

import httpx2 as httpx  # the SDK's own HTTP stack (openai >= 3)
import openai
from openai import OpenAI

MODE = os.environ.get('OAC_MODE', 'controller')
URL = os.environ.get('OAC_URL', '').rstrip('/')
TOOLS_URL = os.environ.get('OAC_TOOLS_URL', '').rstrip('/')
IDLE_URL = os.environ.get('OAC_IDLE_URL', '').rstrip('/')
ENGINE_URL = os.environ.get('OAC_ENGINE_URL', '').rstrip('/')  # the engine behind OAC_URL (paired checks)
SLOW = os.environ.get('OAC_SLOW') == '1'
LIMITED_MODEL = os.environ.get('OAC_DEFAULT_MODEL', '')
N_CTX = int(os.environ.get('OAC_N_CTX', '32768'))
KEY = 'contract-no-key'
USER = [{'role': 'user', 'content': 'Say hello.'}]
TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather for a city.',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}},
         {'type': 'function', 'function': {'name': 'get_time', 'description': 'Time in a zone.',
          'parameters': {'type': 'object', 'properties': {'zone': {'type': 'string'}}, 'required': ['zone']}}}]


def modes(*allowed):
    """Run the test only in these modes (tests.txt pins the same)."""
    def wrap(fn):
        def run(self, *a, **k):
            if MODE not in allowed:
                self.skipTest('mode %s: not applicable' % MODE)
            return fn(self, *a, **k)
        run.__name__, run.__doc__ = fn.__name__, fn.__doc__
        return run
    return wrap


def client(url=None, **kw):
    kw.setdefault('max_retries', 0)
    kw.setdefault('timeout', 60)
    return OpenAI(base_url=(url or URL) + '/v1', api_key=KEY, **kw)


def tokenize(url, text):
    request = urllib.request.Request(url + '/tokenize', data=json.dumps({'content': text, 'add_special': False}).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())['tokens']


def raw_get(path, url=None):
    with urllib.request.urlopen((url or URL) + path, timeout=30) as response:
        return json.loads(response.read())


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = client()
        cls.tools = client(TOOLS_URL or URL)
        models = cls.c.models.list().data
        cls.model = models[0].id

    # ------------------------------------------------------------------ models
    def test_models_list(self):
        page = self.c.models.list()
        self.assertTrue(page.data)
        for row in page.data:
            self.assertEqual(row.object, 'model')
            self.assertIsInstance(row.id, str)

    def test_models_limits(self):
        """Each row publishes its context and its output default (v0.2.0 M1/M2)."""
        for row in self.c.models.list().data:
            extra = row.model_extra or {}
            meta = extra.get('meta') if isinstance(extra.get('meta'), dict) else {}
            limit = extra.get('max_output_tokens', meta.get('max_output_tokens'))
            context = extra.get('context_length', meta.get('n_ctx'))
            self.assertIsInstance(limit, int, row.id)
            self.assertIsInstance(context, int, row.id)
            self.assertLessEqual(limit, context)

    def test_models_retrieve(self):
        row = self.c.models.retrieve(self.model)
        self.assertEqual(row.id, self.model)
        with self.assertRaises(openai.NotFoundError) as caught:
            self.c.models.retrieve('no-such-model-here')
        self.assertEqual(caught.exception.code, 'model_not_found')

    # ------------------------------------------------------------------ chat
    def test_chat_non_stream(self):
        r = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=16)
        self.assertEqual(r.object, 'chat.completion')
        self.assertIsInstance(r.choices[0].message.content, str)
        self.assertIn(r.choices[0].finish_reason, ('stop', 'length'))
        self.assertEqual(r.usage.total_tokens, r.usage.prompt_tokens + r.usage.completion_tokens)

    def test_chat_stream_include_usage(self):
        chunks = list(self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=16, stream=True,
                                                     stream_options={'include_usage': True}))
        text = ''.join(c.choices[0].delta.content or '' for c in chunks if c.choices)
        self.assertTrue(text)
        finish = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
        self.assertEqual(len(finish), 1)
        self.assertEqual(chunks[-1].choices, [])
        self.assertGreater(chunks[-1].usage.completion_tokens, 0)

    def test_usage_details(self):
        """usage.prompt_tokens_details.cached_tokens and completion_tokens_details.reasoning_tokens (S2)."""
        r = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=16)
        self.assertIsInstance(r.usage.prompt_tokens_details.cached_tokens, int)
        self.assertIsInstance(r.usage.completion_tokens_details.reasoning_tokens, int)
        chunks = list(self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=16, stream=True,
                                                     stream_options={'include_usage': True}))
        self.assertIsInstance(chunks[-1].usage.completion_tokens_details.reasoning_tokens, int)

    def test_length_fields(self):
        for kw in ({'max_tokens': 2}, {'max_completion_tokens': 2}):
            r = self.c.chat.completions.create(model=self.model, messages=USER, **kw)
            self.assertEqual(r.choices[0].finish_reason, 'length', kw)
            self.assertLessEqual(r.usage.completion_tokens, 2, kw)

    @modes('controller')
    def test_default_output_limit(self):
        """No length field: the reply ends at the model's default with finish_reason length."""
        row = self.c.models.retrieve(LIMITED_MODEL)
        self.assertEqual(row.model_extra.get('max_output_tokens'), 4)
        r = self.c.chat.completions.create(model=LIMITED_MODEL, messages=USER)
        self.assertEqual((r.choices[0].finish_reason, r.usage.completion_tokens), ('length', 4))

    def test_stop_and_seed(self):
        full = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=32, seed=7).choices[0].message.content
        words = full.split()
        self.assertGreater(len(words), 2)
        cut = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=32, stop=[words[1]])
        self.assertNotIn(words[1], cut.choices[0].message.content)
        self.assertEqual(cut.choices[0].finish_reason, 'stop')
        again = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=32, seed=7).choices[0].message.content
        self.assertEqual(full, again)

    def test_developer_role_and_trailing_assistant(self):
        r = self.c.chat.completions.create(model=self.model, max_tokens=8, messages=[
            {'role': 'developer', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello.'}])
        self.assertEqual(r.object, 'chat.completion')

    def test_image_part(self):
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is this?'},
                                                 {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}}]}]
        if MODE == 'engine':  # the text-only engine refuses it; the controller replaces it by a note
            with self.assertRaises(openai.BadRequestError):
                self.c.chat.completions.create(model=self.model, messages=messages, max_tokens=4)
        else:
            self.assertTrue(self.c.chat.completions.create(model=self.model, messages=messages, max_tokens=4).choices)

    def test_reasoning_effort(self):
        """none/minimal: thinking off (the stub's reply is content); low/medium/high: thinking on (the
        template opens the think block, so the stub's reply is reasoning)."""
        for effort, thinking in (('none', False), ('minimal', False), ('low', True), ('medium', True), ('high', True)):
            r = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=8, reasoning_effort=effort)
            message = r.choices[0].message
            reasoning = (message.model_extra or {}).get('reasoning_content')
            if thinking:
                self.assertTrue(reasoning, effort)
                self.assertGreater(r.usage.completion_tokens_details.reasoning_tokens, 0, effort)
            else:
                self.assertFalse(reasoning, effort)
                self.assertTrue(message.content, effort)
                self.assertEqual(r.usage.completion_tokens_details.reasoning_tokens, 0, effort)

    def test_logprobs(self):
        r = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=4, logprobs=True, top_logprobs=2)
        content = r.choices[0].logprobs.content
        self.assertTrue(content)
        self.assertIsInstance(content[0].logprob, float)

    def test_logit_bias_choice_and_logprobs(self):
        """A logit_bias that bans the stub's scripted first token picks its runner-up; the logprobs
        (non-streamed and streamed) come from the adjusted row. Controller mode: byte-identical
        choice and logprobs to the engine port (the controller forwards the controls unchanged)."""
        probe = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=1, logprobs=True, top_logprobs=2,
                                               temperature=0)
        scripted = probe.choices[0].logprobs.content[0].token
        banned = tokenize(ENGINE_URL or URL, scripted)[0]

        def run(api, stream):
            kw = dict(model=self.model, messages=USER, max_tokens=1, logprobs=True, top_logprobs=2, temperature=0,
                      logit_bias={str(banned): -100}, presence_penalty=0.5, frequency_penalty=0.25)
            if not stream:
                rows = api.chat.completions.create(**kw).choices[0].logprobs.content
            else:
                rows = [row for chunk in api.chat.completions.create(stream=True, **kw)
                        for choice in chunk.choices if choice.logprobs for row in choice.logprobs.content or ()]
            return [(r.token, r.logprob, [(t.token, t.logprob) for t in r.top_logprobs]) for r in rows]

        plain = run(self.c, False)
        self.assertEqual(len(plain), 1)
        self.assertNotEqual(plain[0][0], scripted)  # the banned token was not chosen
        self.assertNotIn(scripted, [t for t, _ in plain[0][2]])  # nor offered: the adjusted row
        self.assertEqual(run(self.c, True), plain)
        if MODE == 'controller' and ENGINE_URL:
            direct = client(ENGINE_URL)
            self.assertEqual(run(direct, False), plain)
            self.assertEqual(run(direct, True), plain)

    @modes('controller')
    def test_prompt_cache_key(self):
        """prompt_cache_key is the session hint: two conversations with the same key (and an
        X-Session-Id with that value) are one session in the ledger; without it each is its own."""
        key = 'contract-' + uuid.uuid4().hex
        seen = {row.get('id') for row in raw_get('/_local/ledger?limit=200')['rows']}
        def ask(text, **kw):
            self.c.chat.completions.create(model=self.model, max_tokens=2, messages=[{'role': 'user', 'content': text + key}], **kw)
        ask('alpha ', prompt_cache_key=key)
        ask('beta ', prompt_cache_key=key)
        ask('gamma ', extra_headers={'X-Session-Id': key})
        ask('delta ')
        rows = [row for row in raw_get('/_local/ledger?limit=200')['rows'] if row.get('id') not in seen]
        self.assertEqual([row.get('path') for row in rows], ['/v1/chat/completions'] * 4)  # exactly these four, newest first
        delta, gamma, beta, alpha = [row.get('session') for row in rows]
        self.assertIsInstance(alpha, str)
        self.assertEqual((beta, gamma), (alpha, alpha))
        self.assertNotEqual(delta, alpha)

    def test_n_greater_than_one(self):
        with self.assertRaises(openai.BadRequestError) as caught:
            self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=4, n=2)
        self.assertEqual(caught.exception.param, 'n')

    @modes('controller')
    def test_response_format(self):
        raw = self.c.chat.completions.with_raw_response.create(model=self.model, messages=USER, max_tokens=16,
                                                               response_format={'type': 'json_object'})
        self.assertEqual(raw.status_code, 200)  # best effort: the stub's prose is returned with a warning
        self.assertIn('JSON', raw.headers.get('x-local-ai-warning', ''))
        r = self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=16, response_format={
            'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'type': 'object'}, 'strict': False}})
        self.assertEqual(r.object, 'chat.completion')
        with self.assertRaises(openai.BadRequestError) as caught:
            self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=4, response_format={
                'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'type': 'object'}, 'strict': True}})
        self.assertEqual(caught.exception.param, 'response_format.json_schema.strict')

    # ------------------------------------------------------------------ tools
    def test_tools_round_trip(self):
        r = self.tools.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': 'Weather and time?'}],
                                               tools=TOOLS, max_tokens=128)
        choice = r.choices[0]
        self.assertEqual(choice.finish_reason, 'tool_calls')
        calls = choice.message.tool_calls
        self.assertEqual([c.function.name for c in calls], ['get_weather', 'get_time'])  # parallel calls
        for call in calls:
            self.assertIsInstance(json.loads(call.function.arguments), dict)
            self.assertTrue(call.id)
        follow = [{'role': 'user', 'content': 'Weather and time?'},
                  {'role': 'assistant', 'content': choice.message.content or '', 'tool_calls': [c.model_dump() for c in calls]}]
        follow += [{'role': 'tool', 'tool_call_id': c.id, 'content': 'ok'} for c in calls]
        again = self.tools.chat.completions.create(model=self.model, messages=follow, tools=TOOLS, max_tokens=128)
        self.assertTrue(again.choices)

    def test_tool_choice_forms(self):
        for choice in ('auto', 'required', 'none', {'type': 'function', 'function': {'name': 'get_time'}}):
            r = self.tools.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': 'Time?'}],
                                                   tools=TOOLS, tool_choice=choice, max_tokens=128)
            self.assertTrue(r.choices, choice)
            if choice == 'required':
                self.assertTrue(r.choices[0].message.tool_calls)
        r = self.tools.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': 'Time?'}],
                                               tools=TOOLS, parallel_tool_calls=False, max_tokens=128)
        self.assertGreaterEqual(len(r.choices[0].message.tool_calls or []), 1)

    def test_tools_stream(self):
        calls, finish = {}, None
        for chunk in self.tools.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': 'Weather?'}],
                                                        tools=TOOLS, max_tokens=128, stream=True):
            for choice in chunk.choices:
                for delta in choice.delta.tool_calls or ():
                    entry = calls.setdefault(delta.index, {'name': '', 'arguments': ''})
                    entry['name'] += (delta.function.name or '') if delta.function else ''
                    entry['arguments'] += (delta.function.arguments or '') if delta.function else ''
                finish = choice.finish_reason or finish
        self.assertEqual(finish, 'tool_calls')
        self.assertEqual([calls[i]['name'] for i in sorted(calls)], ['get_weather', 'get_time'])
        for entry in calls.values():
            self.assertIsInstance(json.loads(entry['arguments']), dict)

    # ------------------------------------------------------------------ errors and headers
    def test_error_shape(self):
        with self.assertRaises(openai.BadRequestError) as caught:  # out of range (-2..2): both servers refuse it
            self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=4, presence_penalty=5)
        body = caught.exception.body  # the error object as sent (the SDK may coerce .code)
        self.assertIsInstance(body, dict)
        self.assertIsInstance(body.get('code'), str)
        self.assertIn('param', body)
        self.assertIsInstance(body.get('message'), str)

    @modes('controller')
    def test_unknown_model(self):
        with self.assertRaises(openai.NotFoundError) as caught:
            self.c.chat.completions.create(model='no-such-model-here', messages=USER, max_tokens=4)
        self.assertEqual((caught.exception.code, caught.exception.param), ('model_not_found', 'model'))

    def test_context_overflow(self):
        big = [{'role': 'user', 'content': 'word ' * (N_CTX + 1000)}]
        with self.assertRaises(openai.BadRequestError) as caught:
            self.c.chat.completions.create(model=self.model, messages=big, max_tokens=4)
        self.assertEqual(caught.exception.code, 'context_length_exceeded')

    def test_request_id_header(self):
        raw = self.c.chat.completions.with_raw_response.create(model=self.model, messages=USER, max_tokens=4)
        self.assertTrue(raw.headers.get('x-request-id'))
        self.assertTrue(raw.request_id)
        with self.assertRaises(openai.BadRequestError) as caught:  # a 4xx names its request too
            self.c.chat.completions.create(model=self.model, messages=USER, max_tokens=4, presence_penalty=5)
        self.assertTrue(caught.exception.response.headers.get('x-request-id'))
        self.assertTrue(caught.exception.request_id)

    @modes('controller')
    def test_request_id_on_mid_stream_error(self):
        """A streamed request that fails after the head went out (it waited behind another one, so
        the queue sent early SSE headers; then the engine refused it) still carries x-request-id,
        and the failure is an SSE error event."""
        if not SLOW:
            self.skipTest('needs the slow-prefill stub engine (OAC_SLOW=1)')
        import threading
        occupant = threading.Thread(target=lambda: list(client().chat.completions.create(
            model=self.model, messages=[{'role': 'user', 'content': 'word ' * 1300}], max_tokens=4, stream=True)))
        occupant.start()
        time.sleep(0.3)
        big = [{'role': 'user', 'content': 'word ' * (N_CTX + 1000)}]
        with self.c.chat.completions.with_streaming_response.create(model=self.model, messages=big, max_tokens=4,
                                                                    stream=True) as response:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers.get('x-request-id'))
            text = '\n'.join(response.iter_lines())
        occupant.join(30)
        self.assertIn('event: error', text)
        self.assertIn('context_length_exceeded', text)

    def test_unsupported_endpoints(self):
        """Embeddings, audio and images answer 404 (never a 5xx the SDK would retry)."""
        idle = client(IDLE_URL) if MODE == 'controller' and IDLE_URL else self.c
        before = raw_get('/_local/status', IDLE_URL)['state'] if idle is not self.c else None
        with self.assertRaises(openai.NotFoundError):
            idle.embeddings.create(model=self.model, input='x')
        with self.assertRaises(openai.NotFoundError):
            idle.audio.speech.create(model=self.model, voice='alloy', input='x')
        with self.assertRaises(openai.NotFoundError):
            idle.images.generate(model=self.model, prompt='x')
        with self.assertRaises(openai.NotFoundError):
            idle.files.list()
        if before is not None:  # controller: none of them loaded a model
            self.assertEqual((before, raw_get('/_local/status', IDLE_URL)['state']), ('unloaded', 'unloaded'))

    # ------------------------------------------------------------------ legacy completions
    def test_legacy_completions(self):
        r = self.c.completions.create(model=self.model, prompt='Say hi', max_tokens=4)
        self.assertEqual(r.object, 'text_completion')
        self.assertIsInstance(r.choices[0].text, str)
        self.assertGreater(r.usage.completion_tokens, 0)
        chunks = list(self.c.completions.create(model=self.model, prompt='Say hi', max_tokens=4, stream=True))
        self.assertTrue(any(c.choices and c.choices[0].text for c in chunks))

    def test_legacy_completions_logprobs_zero(self):
        """An explicit logprobs=0 is not an absent field: the chosen tokens' logprobs come back with
        empty top lists (non-streamed and streamed); without the field there is no logprobs object."""
        r = self.c.completions.create(model=self.model, prompt='Say hi', max_tokens=4, logprobs=0)
        lp = r.choices[0].logprobs
        self.assertIsNotNone(lp)
        self.assertTrue(lp.tokens)
        self.assertEqual(len(lp.token_logprobs), len(lp.tokens))
        self.assertTrue(all(isinstance(x, float) for x in lp.token_logprobs))
        self.assertEqual(lp.top_logprobs, [{} for _ in lp.tokens])
        chunks = list(self.c.completions.create(model=self.model, prompt='Say hi', max_tokens=4, logprobs=0, stream=True))
        rows = [c.choices[0].logprobs for c in chunks if c.choices and c.choices[0].logprobs]
        self.assertTrue(rows)
        self.assertTrue(all(isinstance(x, float) for row in rows for x in row.token_logprobs))
        self.assertTrue(all(top == {} for row in rows for top in row.top_logprobs))
        plain = self.c.completions.create(model=self.model, prompt='Say hi', max_tokens=4)
        self.assertIsNone(plain.choices[0].logprobs)

    # ------------------------------------------------------------------ responses
    @modes('controller')
    def test_responses_basic(self):
        r = self.c.responses.create(model=self.model, input='Say hello.', instructions='Be brief.', max_output_tokens=16)
        self.assertEqual(r.object, 'response')
        self.assertIn(r.status, ('completed', 'incomplete'))
        self.assertTrue(r.output_text)
        self.assertGreater(r.usage.output_tokens, 0)
        self.assertIsInstance(r.usage.input_tokens_details.cached_tokens, int)
        self.assertIsInstance(r.usage.output_tokens_details.reasoning_tokens, int)

    @modes('controller')
    def test_responses_stream(self):
        events = list(self.c.responses.create(model=self.model, input='Say hello.', stream=True))
        kinds = [e.type for e in events]
        self.assertEqual(kinds[:2], ['response.created', 'response.in_progress'])
        self.assertIn('response.output_text.delta', kinds)
        self.assertIn(kinds[-1], ('response.completed', 'response.incomplete'))
        self.assertTrue(events[-1].response.output_text)
        self.assertEqual([e.sequence_number for e in events], list(range(len(events))))

    @modes('controller')
    def test_responses_tools(self):
        tools = [{'type': 'function', 'name': t['function']['name'], 'parameters': t['function']['parameters']} for t in TOOLS]
        r = self.tools.responses.create(model=self.model, input='Weather and time?', tools=tools)
        calls = [item for item in r.output if item.type == 'function_call']
        self.assertEqual([c.name for c in calls], ['get_weather', 'get_time'])
        follow = [{'role': 'user', 'content': 'Weather and time?'}]
        follow += [{'type': 'function_call', 'call_id': c.call_id, 'name': c.name, 'arguments': c.arguments} for c in calls]
        follow += [{'type': 'function_call_output', 'call_id': c.call_id, 'output': 'ok'} for c in calls]
        again = self.tools.responses.create(model=self.model, input=follow, tools=tools)
        self.assertEqual(again.object, 'response')
        streamed = list(self.tools.responses.create(model=self.model, input='Weather?', tools=tools, stream=True))
        kinds = [e.type for e in streamed]
        self.assertIn('response.function_call_arguments.delta', kinds)
        self.assertEqual(kinds[-1], 'response.completed')

    @modes('controller')
    def test_responses_stateful_refused(self):
        with self.assertRaises(openai.BadRequestError) as caught:
            self.c.responses.create(model=self.model, input='x', previous_response_id='resp_123')
        self.assertEqual(caught.exception.param, 'previous_response_id')

    # ------------------------------------------------------------------ retries, queue, replay
    @modes('controller')
    def test_sdk_timeout_retry(self):
        """A non-streamed request that waits behind a long prefill: the SDK times out and retries
        with the same Idempotency-Key; the cut waiter leaves the queue and the retry is served."""
        if not SLOW:
            self.skipTest('needs the slow-prefill stub engine (OAC_SLOW=1)')
        import threading
        long_prompt = [{'role': 'user', 'content': 'word ' * 1300}]
        occupant = threading.Thread(target=lambda: list(client().chat.completions.create(
            model=self.model, messages=long_prompt, max_tokens=4, stream=True)))
        occupant.start()
        time.sleep(0.3)  # the occupant holds the slot (its prefill takes ~2.6 s)
        key = 'contract-' + uuid.uuid4().hex
        retrying = client(timeout=1.5, max_retries=4)
        marker = 'Retry marker %s.' % key[-8:]
        r = retrying.chat.completions.create(model=self.model, messages=[{'role': 'user', 'content': marker}], max_tokens=4,
                                             extra_headers={'Idempotency-Key': key})
        occupant.join(30)
        self.assertEqual(r.object, 'chat.completion')
        rows = [row for row in raw_get('/_local/ledger?limit=50')['rows'] if row.get('path') == '/v1/chat/completions']
        gone = [row for row in rows if row.get('error') == 'client_gone_queued']
        self.assertTrue(gone, 'the first attempt did not wait and leave the queue')

    @modes('controller')
    def test_stream_replay_after_cut(self):
        """A streamed request cut during its prefill and re-sent with the same Idempotency-Key is
        served from the replay buffer (the engine ran once)."""
        if not SLOW:
            self.skipTest('needs the slow-prefill stub engine (OAC_SLOW=1)')
        key = 'contract-' + uuid.uuid4().hex
        messages = [{'role': 'user', 'content': 'replay ' * 1200 + key}]
        first = client(timeout=httpx.Timeout(60, read=0.5))
        try:
            stream = first.chat.completions.create(model=self.model, messages=messages, max_tokens=8, stream=True,
                                                   extra_headers={'Idempotency-Key': key})
            stream.close()  # the client leaves before the first data byte
        except openai.APITimeoutError:
            pass
        time.sleep(0.3)
        text = ''.join(c.choices[0].delta.content or '' for c in client().chat.completions.create(
            model=self.model, messages=messages, max_tokens=8, stream=True, extra_headers={'Idempotency-Key': key}) if c.choices)
        self.assertTrue(text)
        labels = [row.get('replay') for row in raw_get('/_local/ledger?limit=20')['rows']]
        self.assertIn('attached', labels)


if __name__ == '__main__':
    unittest.main()
