use crate::tokenizer::Tokenizer;

pub struct StopPattern {
    pattern: Vec<u32>,
    match_elements_count: usize,
    // Precomputed fallback positions for self-overlapping patterns
    failure_table: Vec<usize>,
}

impl StopPattern {
    pub fn new(pattern: Vec<u32>) -> Self {
        let mut failure_table = vec![0; pattern.len()];
        let mut j = 0;

        // Build the KMP failure table
        for i in 1..pattern.len() {
            while j > 0 && pattern[i] != pattern[j] {
                j = failure_table[j - 1];
            }
            if pattern[i] == pattern[j] {
                j += 1;
            }
            failure_table[i] = j;
        }

        Self {
            pattern,
            match_elements_count: 0,
            failure_table,
        }
    }

    pub fn advance_and_match(&mut self, token: u32) -> bool {
        if self.pattern.is_empty() {
            return false;
        }

        // Fall back through the failure table while tokens don't match
        while self.match_elements_count > 0 && self.pattern[self.match_elements_count] != token {
            self.match_elements_count = self.failure_table[self.match_elements_count - 1];
        }

        // If it matches, advance the counter
        if self.pattern[self.match_elements_count] == token {
            self.match_elements_count += 1;
        }

        // Check if the entire pattern has been matched
        if self.match_elements_count == self.pattern.len() {
            self.match_elements_count = 0; // Reset for future streams
            return true;
        }

        false
    }
}

pub struct StopPatternMatcher {
    patterns: Option<Vec<StopPattern>>,
}

impl StopPatternMatcher {
    pub fn new(patterns: Option<Vec<String>>, tokenizer: &dyn Tokenizer) -> Self {
        let patterns = patterns.map(|list| {
            list.iter()
                .filter_map(|s| tokenizer.encode(s, false).ok())
                .map(StopPattern::new)
                .collect()
        });

        Self { patterns }
    }

    pub fn matches(&mut self, token: u32) -> bool {
        let Some(ref mut patterns) = self.patterns else {
            return false;
        };

        patterns.iter_mut().any(|p| p.advance_and_match(token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::MockTokenizer;
    use rstest::rstest;

    #[rstest]
    #[case(vec![])]
    #[case(vec![1])]
    #[case(vec![1, 2])]
    #[case(vec![1, 2, 3])]
    fn test_stop_pattern_init(#[case] pattern: Vec<u32>) {
        let _ = StopPattern::new(pattern);
    }

    #[rstest]
    #[case(vec![], vec![], false, vec![])]
    #[case(vec![1], vec![], false, vec![])]
    #[case(vec![], vec![1], false, vec![])]
    #[case(vec![1], vec![1], true, vec![0])]
    #[case(vec![1, 2, 3], vec![1, 2, 3], true, vec![2])]
    #[case(vec![1, 2], vec![1, 2], true, vec![1])]
    #[case(vec![1, 2], vec![1, 1, 1, 2], true, vec![3])]
    #[case(vec![1, 2, 3], vec![1, 2, 6, 1, 2, 3], true, vec![5])]
    #[case(vec![1, 1, 1], vec![1, 1, 1, 1], true, vec![2])]
    #[case(vec![1, 2, 3], vec![1, 2, 3, 1, 2, 3], true, vec![2, 5])]
    fn test_stop_pattern_advance_and_match(
        #[case] pattern: Vec<u32>,
        #[case] tokens: Vec<u32>,
        #[case] expected: bool,
        #[case] matched_pos_expected: Vec<usize>,
    ) {
        let mut stop_pattern = StopPattern::new(pattern);

        let mut matched_positions = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let matched = stop_pattern.advance_and_match(*t);
            if matched {
                matched_positions.push(i);
            }
        }
        let matched = !matched_positions.is_empty();
        assert_eq!(matched, expected);
        assert_eq!(matched_positions, matched_pos_expected);
    }

    #[rstest]
    #[case(vec![], 0, vec![], vec![], false)]
    #[case(vec!["hello".to_string()], 1, vec![1], vec![1], true)]
    #[case(vec!["hello".to_string(), "world".to_string()], 2, vec![1, 2], vec![1, 2], true)]
    #[case(vec!["hello".to_string()], 1, vec![1], vec![1, 2], true)]
    #[case(vec!["hello".to_string(), "world".to_string()], 2, vec![1, 2], vec![1], false)]
    #[case(vec!["hello".to_string(), "world".to_string()], 2, vec![1, 2, 3], vec![1, 2], false)]
    #[case(vec!["hello".to_string(), "world".to_string()], 2, vec![1, 2], vec![1, 2, 3], true)]
    fn test_stop_pattern_matcher(
        #[case] patterns: Vec<String>,
        #[case] tokenizer_call_count: usize,
        #[case] encode_result: Vec<u32>,
        #[case] tokens: Vec<u32>,
        #[case] expected_matched: bool,
    ) {
        let mut mock_tokenizer = MockTokenizer::new();
        mock_tokenizer
            .expect_encode()
            .times(tokenizer_call_count)
            .returning(move |_, _| Ok(encode_result.clone()));

        let mut matcher = StopPatternMatcher::new(Some(patterns), &mock_tokenizer);

        let mut matched = false;
        for token in tokens {
            matched = matcher.matches(token);
            if matched {
                break;
            }
        }

        assert_eq!(matched, expected_matched);
    }
}
