// Content Credentials (C2PA): signed provenance manifests that Adobe, OpenAI,
// Google, Microsoft, camera makers and others embed in images. Read with the
// official C2PA web SDK, loaded only when a file carries a manifest.

let sdk = null;

function load() {
  if (!sdk) {
    sdk = (async () => {
      const mod = await import(new URL('../vendor/c2pa/c2pa.js', import.meta.url).href);
      const c2pa = await mod.createC2pa({ wasmSrc: new URL('../vendor/c2pa/c2pa_bg.wasm', import.meta.url).href });
      return { mod, c2pa };
    })();
    sdk.catch(() => {
      sdk = null;
    });
  }
  return sdk;
}

/** The manifest store of a file, or null if it has none. */
export async function readCredentials(file) {
  const { mod, c2pa } = await load();
  const reader = await mod.Reader.fromBlob(c2pa, file.type || 'image/jpeg', file);
  if (!reader) return null;
  try {
    return await reader.manifestStore();
  } finally {
    await reader.free?.();
  }
}

const AI_TYPES = /trainedAlgorithmicMedia|algorithmicMedia/;

/** What a manifest store says, in plain terms. */
export function summarize(store) {
  const m = store?.manifests?.[store.active_manifest];
  if (!m) return null;
  const assertions = m.assertions || [];
  const actions = assertions.filter((a) => /^c2pa\.actions/.test(a.label)).flatMap((a) => a.data?.actions || []);
  const sourceTypes = actions.map((a) => a.digitalSourceType || a.parameters?.digitalSourceType).filter(Boolean);
  const work = assertions.find((a) => /CreativeWork/.test(a.label))?.data;
  const authors = (work?.author || []).map((x) => x.name).filter(Boolean);
  const failures = store.validation_results?.activeManifest?.failure || (store.validation_status || []).filter((v) => !/success|validated|match/i.test(v.code));
  return {
    label: store.active_manifest,
    labels: Object.keys(store.manifests || {}),
    instanceId: m.instance_id,
    title: m.title,
    generator: m.claim_generator_info?.[0]?.name || m.claim_generator,
    signer: m.signature_info?.issuer || m.signature_info?.common_name,
    time: m.signature_info?.time,
    actions: actions.map((a) => a.action.replace(/^c2pa\./, '').replace(/_/g, ' ')),
    ai: sourceTypes.some((t) => AI_TYPES.test(t)),
    authors,
    ingredients: (m.ingredients || []).map((i) => ({ title: i.title, relationship: i.relationship, manifest: i.active_manifest, instanceId: i.instance_id, documentId: i.document_id })),
    untrusted: failures.some((f) => /untrusted/i.test(f.code)),
    invalid: failures.filter((f) => !/untrusted|ocsp/i.test(f.code)).map((f) => f.explanation || f.code),
  };
}

/** Does `child` say it was made from `parent`? */
export function linked(child, parent) {
  if (!child || !parent) return false;
  if (child.labels.includes(parent.label) && child.label !== parent.label) return true;
  return child.ingredients.some((i) => (i.manifest && i.manifest === parent.label) || (i.instanceId && i.instanceId === parent.instanceId));
}

export function describe(s) {
  if (!s) return 'no Content Credentials';
  const parts = [
    s.generator && `made with ${s.generator}`,
    s.signer && `signed by ${s.signer}${s.untrusted ? ' (not on the trust list)' : ''}${s.time ? ` on ${s.time.slice(0, 10)}` : ''}`,
    s.actions.length && `actions: ${s.actions.join(', ')}`,
    s.ai && 'declares generative AI',
    s.authors.length && `author: ${s.authors.join(', ')}`,
    s.ingredients.length && `ingredients: ${s.ingredients.map((i) => i.title).join(', ')}`,
    s.invalid.length && `validation problems: ${s.invalid.slice(0, 2).join('; ')}`,
  ].filter(Boolean);
  return parts.join(' · ');
}
