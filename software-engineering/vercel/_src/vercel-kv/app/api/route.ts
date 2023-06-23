import { createClient } from '@vercel/kv';
import { NextResponse } from 'next/server';
 
export async function GET() {
  const users = createClient({
    url: process.env.KV_REST_API_URL,
    token: process.env.KV_REST_API_TOKEN,
  });
 
  const user = await users.hgetall('user:me');
 
  const products = createClient({
    url: process.env.KV_REST_API_URL,
    token: process.env.KV_REST_API_TOKEN,
  });
 
  const product = await products.hgetall('product:shirt');
 
  return NextResponse.json({ user, product });
}
