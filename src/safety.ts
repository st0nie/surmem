/** Security boundary for durable memory content. */

export interface SafetyFinding {
  id: string;
  category: "secret" | "injection" | "unicode";
  severity: "medium" | "high";
}

const INVISIBLE = /[\u200B-\u200D\u2060\u202A-\u202E\uFEFF]/u;

const RULES: Array<SafetyFinding & { pattern: RegExp }> = [
  {
    id: "prompt_injection",
    category: "injection",
    severity: "high",
    pattern: /ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions/i,
  },
  {
    id: "role_hijack",
    category: "injection",
    severity: "high",
    pattern: /(?:you are now|act as)\s+(?:the\s+)?(?:system|developer|administrator|root)/i,
  },
  {
    id: "hidden_instruction",
    category: "injection",
    severity: "high",
    pattern: /do\s+not\s+(?:tell|show|reveal)\s+(?:the\s+)?user/i,
  },
  {
    id: "system_tag",
    category: "injection",
    severity: "medium",
    pattern: /<\/?(?:system|developer|system-reminder|memory-policy)\b/i,
  },
  {
    id: "exfiltration",
    category: "injection",
    severity: "high",
    pattern: /(?:curl|wget)\s+[^\n]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|\.env)/i,
  },
  {
    id: "private_key",
    category: "secret",
    severity: "high",
    pattern: /-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----/,
  },
  { id: "openai_key", category: "secret", severity: "high", pattern: /\bsk-[A-Za-z0-9_-]{20,}\b/ },
  { id: "anthropic_key", category: "secret", severity: "high", pattern: /\bsk-ant-[A-Za-z0-9_-]{20,}\b/ },
  { id: "github_token", category: "secret", severity: "high", pattern: /\bgh[opsu]_[A-Za-z0-9]{20,}\b/ },
  { id: "aws_key", category: "secret", severity: "high", pattern: /\bAKIA[0-9A-Z]{16}\b/ },
  {
    id: "bearer_token",
    category: "secret",
    severity: "high",
    pattern: /\bBearer\s+[A-Za-z0-9._~+/-]{20,}\b/i,
  },
  {
    id: "jwt",
    category: "secret",
    severity: "high",
    pattern: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/,
  },
  {
    id: "secret_assignment",
    category: "secret",
    severity: "medium",
    pattern: /\b(?:password|passwd|secret|token|api[_-]?key)\s*[:=]\s*["']?[^\s"']{8,}/i,
  },
  {
    id: "credential_env",
    category: "secret",
    severity: "medium",
    pattern: /\b(?:OPENAI|ANTHROPIC|OPENROUTER|GITHUB|AWS)_(?:API_)?(?:KEY|TOKEN|SECRET)\b/,
  },
];

export function scanMemoryContent(text: string): SafetyFinding[] {
  const findings: SafetyFinding[] = [];
  if (INVISIBLE.test(text)) {
    findings.push({ id: "invisible_unicode", category: "unicode", severity: "high" });
  }
  for (const { pattern, ...finding } of RULES) {
    if (pattern.test(text)) findings.push(finding);
  }
  return findings;
}

/** Collapse control characters and instruction-like tags before prompt display. */
export function sanitizeForPrompt(text: string, maxChars = 2000): string {
  return (
    text
      // biome-ignore lint/suspicious/noControlCharactersInRegex: memory input must have controls stripped before prompt use.
      .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, " ")
      .replace(/<\/?(?:system|developer|system-reminder|memory-policy)\b[^>]*>/gi, "")
      .replace(/\r\n?/g, "\n")
      .trim()
      .slice(0, Math.max(0, maxChars))
  );
}

export function escapeXmlData(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
