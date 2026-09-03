import { modelPrompt, normalizeCoachRequest, rulesResponse } from './coach';

type ModelResponse={output_text?:unknown;output?:Array<{content?:Array<{type?:string;text?:string}>}>;message?:{content?:string}};
function extractResponseText(data: ModelResponse): string {
  if (typeof data.output_text === 'string') return data.output_text;
  return (data.output || []).flatMap(item => item.content || []).find(item => item.type === 'output_text')?.text || '';
}

async function askConfiguredModel(prompt: string): Promise<{ answer:string; mode:'ollama'|'huggingface'|'openai' } | null> {
  const ollamaBase = process.env.OLLAMA_BASE_URL?.replace(/\/$/, '');
  if (ollamaBase) {
    const response = await fetch(`${ollamaBase}/api/chat`, { method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({model:process.env.OLLAMA_MODEL || 'qwen2.5:7b', stream:false, messages:[{role:'user',content:prompt}]}), signal:AbortSignal.timeout(20000) });
    if (!response.ok) throw new Error('The local model was unavailable.');
    const data = await response.json() as ModelResponse;
    if (data?.message?.content) return {answer:data.message.content,mode:'ollama'};
  }
  if (process.env.HF_TOKEN) {
    const response = await fetch('https://router.huggingface.co/v1/responses', { method:'POST', headers:{'content-type':'application/json',authorization:`Bearer ${process.env.HF_TOKEN}`}, body:JSON.stringify({model:process.env.HF_MODEL || 'openai/gpt-oss-120b:fastest',input:prompt}), signal:AbortSignal.timeout(20000) });
    if (!response.ok) throw new Error('The Hugging Face model was unavailable.');
    const answer = extractResponseText(await response.json() as ModelResponse);
    if (answer) return {answer,mode:'huggingface'};
  }
  if (process.env.OPENAI_API_KEY) {
    const base = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/$/, '');
    const response = await fetch(`${base}/responses`, { method:'POST', headers:{'content-type':'application/json',authorization:`Bearer ${process.env.OPENAI_API_KEY}`}, body:JSON.stringify({model:process.env.OPENAI_MODEL || 'gpt-5-mini',input:prompt}), signal:AbortSignal.timeout(20000) });
    if (!response.ok) throw new Error('The hosted model was unavailable.');
    const answer = extractResponseText(await response.json());
    if (answer) return {answer,mode:'openai'};
  }
  return null;
}

export async function POST(request: Request) {
  try {
    const input = normalizeCoachRequest(await request.json());
    const fallback = rulesResponse(input);
    try {
      const enhanced = await askConfiguredModel(modelPrompt(input, fallback));
      if (enhanced) return Response.json({...fallback,...enhanced});
    } catch { /* Keep the experience available when an optional provider fails. */ }
    return Response.json({...fallback,mode:'rules'});
  } catch (error) {
    return Response.json({error:error instanceof Error?error.message:'Invalid request.'},{status:400});
  }
}
