import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeCoachRequest, rulesResponse } from './coach.ts';

test('coach limits untrusted input and evidence',()=>{const result=normalizeCoachRequest({message:'x'.repeat(900),evidence:Array.from({length:8},(_,i)=>({field:`Field ${i}`}))});assert.equal(result.message.length,800);assert.equal(result.evidence?.length,5)});
test('coach grounds the fallback in supplied evidence',()=>{const result=rulesResponse({message:'why?',evidence:[{field:'Computer Science',credential:"Bachelor's",jobGrowth:12,programChange:5}]});assert.match(result.answer,/Computer Science/);assert.match(result.answer,/\+12.0%/);assert.match(result.answer,/not a prediction/)});
test('coach handles an empty evidence state',()=>{assert.match(rulesResponse({message:'help'}).answer,/interest or choose a work style/)});
