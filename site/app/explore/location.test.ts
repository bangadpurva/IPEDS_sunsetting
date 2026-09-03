import assert from 'node:assert/strict';
import test from 'node:test';
import {haversineMiles,parsePlaceInput,resolveUsPlace} from './location.ts';

test('parses ZIP and city/state input',()=>{
  assert.deepEqual(parsePlaceInput('48201'),{kind:'zip',zip:'48201'});
  assert.deepEqual(parsePlaceInput('Detroit, MI'),{kind:'city',city:'Detroit',state:'mi'});
  assert.equal(parsePlaceInput('Detroit'),null);
});

test('calculates a plausible Detroit to Ann Arbor distance',()=>{
  const miles=haversineMiles({latitude:42.3314,longitude:-83.0458},{latitude:42.2808,longitude:-83.7430});
  assert.ok(miles>34&&miles<38);
});

test('resolves a ZIP through the public lookup response',async()=>{
  const fakeFetch=async()=>new Response(JSON.stringify({places:[{'place name':'Detroit','state abbreviation':'MI',latitude:'42.33',longitude:'-83.05'}]}),{status:200});
  assert.deepEqual(await resolveUsPlace('48201',fakeFetch),{latitude:42.33,longitude:-83.05,label:'Detroit, MI',state:'MI'});
});

test('returns a useful error for unsupported input',async()=>{
  await assert.rejects(()=>resolveUsPlace('Detroit',async()=>new Response()),/5-digit ZIP/);
});
