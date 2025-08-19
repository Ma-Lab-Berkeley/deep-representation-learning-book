// Netlify serverless function to proxy chat requests to an OpenAI-compatible API
// Expects env vars: OPENAI_API_KEY (required), OPENAI_BASE_URL (optional)

export async function handler(event) {
  try {
    if (event.httpMethod !== 'POST') {
      return { statusCode: 405, body: JSON.stringify({ error: 'Method Not Allowed' }) };
    }

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return { statusCode: 500, body: JSON.stringify({ error: 'Server not configured: missing OPENAI_API_KEY' }) };
    }

    const baseUrl = (process.env.OPENAI_BASE_URL || 'https://api.openai.com').replace(/\/$/, '');
    const url = `${baseUrl}/v1/chat/completions`;

    const body = event.body ? JSON.parse(event.body) : {};
    // Basic input validation
    if (!body || !Array.isArray(body.messages)) {
      return { statusCode: 400, body: JSON.stringify({ error: 'Invalid request: messages array required' }) };
    }

    const fetchResponse = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: body.model || 'gpt-4o-mini',
        messages: body.messages,
        temperature: typeof body.temperature === 'number' ? body.temperature : 0.2,
        stream: false
      })
    });

    const text = await fetchResponse.text();
    const status = fetchResponse.status;

    // Pass through JSON if possible; otherwise, wrap
    let payload;
    try { payload = JSON.parse(text); }
    catch { payload = { error: 'Upstream returned non-JSON', raw: text } }

    return {
      statusCode: status,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    };
  } catch (err) {
    return { statusCode: 500, body: JSON.stringify({ error: err && err.message ? err.message : String(err) }) };
  }
}


