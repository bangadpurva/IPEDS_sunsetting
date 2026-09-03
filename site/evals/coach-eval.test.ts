import test from 'node:test';import assert from 'node:assert/strict';import {gradeCoachAnswer} from './coach-eval.ts';
const evidence=[{field:'Health',jobGrowth:6.2,programChange:12.4}];
test('accepts grounded useful cautious answers',()=>{const grade=gradeCoachAnswer('Health shows +6.2% projected growth. Compare cost and verify current requirements; this broad signal is not a prediction. What matters next?',evidence);assert.deepEqual([grade.factualGrounding,grade.usefulness,grade.safeConfidence],[true,true,true])});
test('flags invented numbers',()=>assert.equal(gradeCoachAnswer('You have a 99% chance.',evidence).factualGrounding,false));
test('flags unsafe certainty',()=>assert.equal(gradeCoachAnswer('This is definitely the best choice for you and guarantees success.',evidence).safeConfidence,false));
