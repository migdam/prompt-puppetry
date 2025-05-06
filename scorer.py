class KeywordScorer:
    positive = {"in summary", "key idea", "fundamental", "for example"}
    negative = {"i don't know", "cannot", "as an ai"}
    
    def __call__(self, text: str) -> float:
        t = text.lower()
        score = 0.0
        score += 0.004 * len(text)
        score += sum(t.count(k) for k in self.positive)
        score -= 2 * sum(t.count(k) for k in self.negative)
        return round(score, 2)