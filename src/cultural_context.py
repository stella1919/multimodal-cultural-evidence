from typing import Any

def infer_context(record: dict[str, Any]) -> tuple[str, str]:
    fields = " ".join(str(record.get(k, "") or "") for k in ("geography", "country", "region", "culture", "title", "object_name")).lower()
    geography = " ".join(str(record.get(k, "") or "") for k in ("geography", "country", "region")).lower()
    if any(x in geography for x in ("tibet", "tibetan", "himalaya", "nepal", "nepalese", "bhutan")):
        return "Himalayan", "geography matched Himalayan indicators"
    if any(x in geography for x in ("china", "chinese", "japan", "japanese", "korea", "korean")) or any(x in fields for x in ("ming", "qing", "tang", "song")):
        return "East_Asia", "geography or period matched East Asia indicators"
    if any(x in geography for x in ("india", "indian", "south asia", "pakistan", "sri lanka")):
        return "South_Asia", "geography matched South Asia indicators"
    return "Other", "insufficient or ambiguous contextual evidence"

