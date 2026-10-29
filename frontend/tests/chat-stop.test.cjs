// Unit-test the chat form's stop/abort state with a minimal DOM and SSE transport.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() { this.style = {}; this.listeners = {}; this.nodes = {}; this.children = []; this.value = ''; this.hidden = true; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  querySelector(name) { return this.nodes[name] ||= new Element(); }
  querySelectorAll() { return []; }
  appendChild(child) { this.children.push(child); }
  remove() {}
  focus() { this.focused = true; }
}
(async () => {
  const form = new Element(), prompt = new Element(), stop = new Element(), send = new Element();
  prompt.value = 'Generate Python code';
  let nextRead, requestBody, stopUrl;
  const encoder = new TextEncoder();
  const context = {
    console, TextDecoder, AbortController, AbortSignal, crypto: require('node:crypto').webcrypto,
    setInterval, clearInterval, setTimeout, clearTimeout,
    chatForm: form, promptInput: prompt, attachedImages: [], currentEngineMode: 'agentic', currentSessionId: null,
    document: {getElementById: id => id === 'stop-chat-btn' ? stop : send, createElement: () => new Element()},
    welcomeHero: new Element(), chatThread: new Element(), localStorage: {removeItem() {}},
    appendMessage() {}, renderAttachedImagesPreview() {}, scrollToBottom() {},
    getPrefillPlaceholderHtml: () => 'Starting...', renderMarkdown: text => text,
    renderThinkingBody: text => text, escapeHtml: text => text, renderAgentTraceStep() {},
    loadSessions: async () => {},
    fetch: async (url, options) => {
      if (url.endsWith('/stop')) {
        stopUrl = url;
        nextRead({done:false, value:encoder.encode('data: {"type":"stopped","session_id":42}\n\n')});
        return {ok:true};
      }
      requestBody = JSON.parse(options.body);
      let first = true;
      return {ok:true, body:{getReader:() => ({read:() => {
        if (options.signal.aborted) return Promise.reject(Object.assign(new Error('aborted'), {name:'AbortError'}));
        if (first) { first=false; return Promise.resolve({done:false, value:encoder.encode('data: {"type":"session","session_id":42}\n\ndata: {"type":"content","delta":"partial code"}\n\n')}); }
        return new Promise(resolve => { nextRead=resolve; });
      }})}};
    },
  };
  vm.createContext(context);
  const source = fs.readFileSync(require('node:path').join(__dirname, '../app.js'), 'utf8');
  vm.runInContext(source.slice(source.indexOf('let activeChat = null;'), source.indexOf('// Message Rendering')), context);
  const pending = form.listeners.submit({preventDefault(){}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(send.disabled, true);
  assert.equal(stop.hidden, false);
  prompt.value = 'Use coordinates without forces';
  await stop.listeners.click();
  await pending;
  assert.equal(stopUrl, `/api/chat/${requestBody.request_id}/stop`);
  assert.equal(send.disabled, false);
  assert.equal(stop.hidden, true);
  assert.equal(prompt.value, 'Use coordinates without forces');
  assert.equal(prompt.focused, true);
  assert.equal(context.currentSessionId, 42);
  console.log('PASS: Stop cancels request, preserves guidance, and restores Send');
})().catch(error => { console.error(error); process.exitCode=1; });
