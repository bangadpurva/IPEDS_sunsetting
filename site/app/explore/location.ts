export type ResolvedLocation = { latitude:number; longitude:number; label:string; state?:string };

export function haversineMiles(from:{latitude:number;longitude:number},to:{latitude:number;longitude:number}){
  const radians=(degrees:number)=>degrees*Math.PI/180;
  const earthRadiusMiles=3958.8;
  const dLat=radians(to.latitude-from.latitude);
  const dLon=radians(to.longitude-from.longitude);
  const a=Math.sin(dLat/2)**2+Math.cos(radians(from.latitude))*Math.cos(radians(to.latitude))*Math.sin(dLon/2)**2;
  return earthRadiusMiles*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}

export function parsePlaceInput(value:string){
  const input=value.trim();
  if(/^\d{5}$/.test(input))return{kind:'zip' as const,zip:input};
  const match=input.match(/^(.+),\s*([A-Za-z]{2})$/);
  if(match)return{kind:'city' as const,city:match[1].trim(),state:match[2].toLowerCase()};
  return null;
}

export async function resolveUsPlace(value:string,fetcher:typeof fetch=fetch):Promise<ResolvedLocation>{
  const parsed=parsePlaceInput(value);
  if(!parsed)throw new Error('Enter a 5-digit ZIP code or a city and two-letter state, such as Detroit, MI.');
  const url=parsed.kind==='zip'?`https://api.zippopotam.us/us/${parsed.zip}`:`https://api.zippopotam.us/us/${parsed.state}/${encodeURIComponent(parsed.city)}`;
  const response=await fetcher(url);
  if(!response.ok)throw new Error('We could not find that U.S. location. Check the spelling or ZIP code.');
  const payload=await response.json();
  const places=Array.isArray(payload)?payload.flatMap(item=>item.places||[]):payload.places||[];
  if(!places.length)throw new Error('We could not find that U.S. location. Check the spelling or ZIP code.');
  const place=places[0];
  return{latitude:Number(place.latitude),longitude:Number(place.longitude),label:`${place['place name']}, ${place['state abbreviation']}`,state:place['state abbreviation']};
}
