"""
Medical Named Entity Recognition (NER) module.
Extracts medical entities from text.
"""
import re
from typing import List, Dict, Set


class MedicalNER:
    """
    Simple rule-based medical entity extraction.
    For production, would use spaCy with medical models or fine-tuned transformers.
    """
    
    def __init__(self):
        # Common medical patterns (simplified for demo)
        self.diagnosis_keywords = [
            'diabetes', 'hypertension', 'depression', 'anxiety', 'cancer',
            'arthritis', 'asthma', 'copd', 'pneumonia', 'covid', 'influenza',
            'fracture', 'injury', 'strain', 'sprain', 'contusion', 'laceration'
        ]
        
        self.medication_patterns = [
            r'\b[A-Z][a-z]+(?:ol|ine|cin|mab|pril|sartan|statin)\b',  # Common drug suffixes
            r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|codeine)\b'
        ]
        
        self.symptom_keywords = [
            'pain', 'fever', 'cough', 'fatigue', 'nausea', 'vomiting',
            'headache', 'dizziness', 'shortness of breath', 'chest pain',
            'back pain', 'abdominal pain', 'swelling', 'rash'
        ]
        
        self.procedure_keywords = [
            'surgery', 'operation', 'biopsy', 'x-ray', 'mri', 'ct scan',
            'ultrasound', 'endoscopy', 'colonoscopy', 'physical therapy'
        ]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        text_lower = text.lower()
        
        entities = {
            'diagnoses': self._extract_diagnoses(text_lower),
            'medications': self._extract_medications(text),
            'symptoms': self._extract_symptoms(text_lower),
            'procedures': self._extract_procedures(text_lower),
            'restrictions': self._extract_restrictions(text_lower)
        }
        
        return entities
    
    def _extract_diagnoses(self, text: str) -> List[str]:
        """Extract diagnoses from text"""
        found = set()
        for diagnosis in self.diagnosis_keywords:
            if diagnosis in text:
                found.add(diagnosis.title())
        return list(found)
    
    def _extract_medications(self, text: str) -> List[str]:
        """Extract medications using pattern matching"""
        found = set()
        for pattern in self.medication_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found.update([m.title() for m in matches])
        return list(found)
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        found = set()
        for symptom in self.symptom_keywords:
            if symptom in text:
                found.add(symptom.title())
        return list(found)
    
    def _extract_procedures(self, text: str) -> List[str]:
        """Extract medical procedures from text"""
        found = set()
        for procedure in self.procedure_keywords:
            if procedure in text:
                found.add(procedure.title())
        return list(found)
    
    def _extract_restrictions(self, text: str) -> List[str]:
        """Extract work restrictions and limitations"""
        restrictions = []
        
        # Look for common restriction patterns
        restriction_patterns = [
            r'no lifting (?:more than )?([\w\s]+)',
            r'(?:unable|not able) to (?:work|perform) ([\w\s]+)',
            r'restricted from ([\w\s]+)',
            r'limited to ([\w\s]+)',
            r'cannot ([\w\s]+)',
            r'avoid ([\w\s]+)'
        ]
        
        for pattern in restriction_patterns:
            matches = re.findall(pattern, text)
            restrictions.extend([f"Restriction: {m}" for m in matches])
        
        return restrictions[:10]  # Limit to 10 restrictions
    
    def extract_timeline_events(self, text: str) -> List[Dict]:
        """
        Extract medical events with dates for timeline.
        Simplified version - production would use more sophisticated date parsing.
        """
        events = []
        
        # Find dates
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        dates = re.findall(date_pattern, text)
        
        # Simple approach: look for sentences with dates
        sentences = text.split('.')
        for sentence in sentences:
            for date in dates:
                if date in sentence:
                    events.append({
                        'date': date,
                        'event': sentence.strip()
                    })
        
        return events[:20]  # Limit to 20 events
    
    def summarize_medical_profile(self, text: str) -> Dict:
        """
        Create a summary of the medical profile from text.
        
        Args:
            text: Full medical document text
            
        Returns:
            Dictionary with summary information
        """
        entities = self.extract_entities(text)
        
        return {
            'total_diagnoses': len(entities['diagnoses']),
            'diagnoses': entities['diagnoses'][:5],  # Top 5
            'total_medications': len(entities['medications']),
            'medications': entities['medications'][:5],  # Top 5
            'key_symptoms': entities['symptoms'][:5],
            'procedures': entities['procedures'][:5],
            'restrictions': entities['restrictions'][:3]
        }
