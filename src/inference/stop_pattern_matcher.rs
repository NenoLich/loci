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
        if self.pattern.is_empty() { return false; }

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
                .map(|ids| StopPattern::new(ids))
                .collect()
        });

        Self { patterns }
    }

    pub fn matches(&mut self, token: u32) -> bool {
        let Some(ref mut patterns) = self.patterns else { return false; };
    
        patterns.iter_mut().any(|p| p.advance_and_match(token))
    }
}