use std::collections::BTreeSet;

pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    for raw in text.split(|ch: char| !is_identifier_char(ch)) {
        if raw.is_empty() {
            continue;
        }

        let normalized = raw.to_ascii_lowercase();
        if normalized.len() > 1 {
            tokens.push(normalized.clone());
        }

        for split in split_identifier(raw) {
            if split.len() > 1 && split != normalized {
                tokens.push(split);
            }
        }
    }

    tokens
}

pub fn unique_tokens(text: &str) -> Vec<String> {
    tokenize(text)
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub fn split_identifier(token: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut previous_was_lower = false;

    for ch in token.chars() {
        if ch == '_' || ch == '-' {
            push_part(&mut parts, &mut current);
            previous_was_lower = false;
            continue;
        }

        if ch.is_ascii_uppercase() && previous_was_lower && !current.is_empty() {
            push_part(&mut parts, &mut current);
        }

        previous_was_lower = ch.is_ascii_lowercase();
        current.push(ch.to_ascii_lowercase());
    }

    push_part(&mut parts, &mut current);
    parts
}

fn push_part(parts: &mut Vec<String>, current: &mut String) {
    if !current.is_empty() {
        parts.push(std::mem::take(current));
    }
}

fn is_identifier_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'
}

#[cfg(test)]
mod tests {
    use super::{split_identifier, tokenize};

    #[test]
    fn splits_snake_and_camel_case() {
        assert_eq!(
            split_identifier("UserSessionStore"),
            vec!["user", "session", "store"]
        );
        assert_eq!(
            split_identifier("user_session_store"),
            vec!["user", "session", "store"]
        );
    }

    #[test]
    fn tokenizes_identifiers_and_words() {
        let tokens = tokenize("pub fn fetchUserProfile(user_id: UserId)");
        assert!(tokens.contains(&"fetchuserprofile".to_string()));
        assert!(tokens.contains(&"fetch".to_string()));
        assert!(tokens.contains(&"profile".to_string()));
        assert!(tokens.contains(&"user".to_string()));
    }
}
