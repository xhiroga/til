import { assertEquals } from "https://deno.land/std@0.161.0/testing/asserts.ts";

Deno.test("Normalize", () => {
    assertEquals('ガ'.normalize('NFC'), 'ガ');
    assertEquals('ガ'.normalize('NFD'), 'ガ');

    assertEquals('ｶﾞ'.normalize('NFKC'), 'ガ');
    assertEquals('ｶﾞ'.normalize('NFKD'), 'ガ');
});
