import logging
import random
from collections import Counter

logger = logging.getLogger(__name__)

def get_recommendations(facial_features, products, max_recommendations=6):
    """
    Generate personalized product recommendations based on enhanced facial analysis
    
    Args:
        facial_features: Dictionary containing facial analysis results
        products: List of Product objects
        max_recommendations: Maximum number of products to recommend
        
    Returns:
        list: List of recommended Product objects with optimized routine
    """
    if not facial_features or not products:
        logger.warning("No facial features or products available for recommendations")
        return []
    
    # Extract enhanced features
    skin_type = facial_features.get('skin_type', 'normal')
    skin_tone = facial_features.get('skin_tone', 'medium-neutral')
    concerns = facial_features.get('concerns', [])
    
    # Log the detected features for debugging
    logger.info(f"Detected skin type: {skin_type}")
    logger.info(f"Detected skin tone: {skin_tone}")
    logger.info(f"Detected concerns: {concerns}")
    
    # Define skin type groups for better matching
    dry_types = ['dry', 'very-dry']
    oily_types = ['oily', 'very-oily']
    sensitive_types = ['sensitive']
    combination_types = ['combination']
    normal_types = ['normal']
    
    # Define skin tone groups
    light_tones = ['fair', 'fair-warm', 'fair-cool', 'fair-neutral', 'light', 'light-warm', 'light-cool', 'light-neutral']
    medium_tones = ['medium', 'medium-warm', 'medium-neutral', 'medium-cool']
    deep_tones = ['deep-warm', 'deep-cool', 'warm-tan', 'neutral-tan']
    
    # Parse skin tone undertones
    warm_undertones = ['fair-warm', 'light-warm', 'medium-warm', 'deep-warm', 'warm-tan']
    cool_undertones = ['fair-cool', 'light-cool', 'medium-cool', 'deep-cool']
    neutral_undertones = ['fair-neutral', 'light-neutral', 'medium-neutral', 'neutral-tan']
    
    # Set user's detected groups
    user_tone_group = 'medium'
    if any(tone in skin_tone for tone in light_tones):
        user_tone_group = 'light'
    elif any(tone in skin_tone for tone in deep_tones):
        user_tone_group = 'deep'
    
    user_undertone = 'neutral'
    if any(tone in skin_tone for tone in warm_undertones):
        user_undertone = 'warm'
    elif any(tone in skin_tone for tone in cool_undertones):
        user_undertone = 'cool'
    
    # Score each product based on how well it matches the facial features
    scored_products = []
    
    # Create a counter to track product categories for balanced routine
    recommended_categories = Counter()
    
    # Category importance based on concerns (prioritize certain product types)
    category_importance = {
        'cleanser': 1.0,      # Base importance
        'toner': 0.8,
        'serum': 1.2,
        'moisturizer': 1.0,
        'suncare': 1.0,
        'mask': 0.5,
        'exfoliant': 0.7,
        'treatment': 0.9,
        'eye care': 0.6,
        'primer': 0.4,
        'tools': 0.3,
        'lip care': 0.3,
        'oil': 0.6
    }
    
    # Adjust category importance based on concerns
    if 'acne' in concerns or 'oiliness' in concerns:
        category_importance['cleanser'] += 0.5
        category_importance['exfoliant'] += 0.4
        category_importance['mask'] += 0.3
        category_importance['oil'] -= 0.3
    
    if 'aging' in concerns or 'wrinkles' in concerns:
        category_importance['serum'] += 0.6
        category_importance['treatment'] += 0.5
        category_importance['eye care'] += 0.7
        category_importance['moisturizer'] += 0.3
    
    if 'dryness' in concerns:
        category_importance['moisturizer'] += 0.6
        category_importance['oil'] += 0.5
        category_importance['serum'] += 0.3
        category_importance['exfoliant'] -= 0.2
    
    if 'dullness' in concerns or 'uneven tone' in concerns:
        category_importance['exfoliant'] += 0.5
        category_importance['serum'] += 0.3
        category_importance['mask'] += 0.2
    
    if 'sensitivity' in concerns or 'redness' in concerns:
        category_importance['treatment'] += 0.4
        category_importance['exfoliant'] -= 0.3
        category_importance['cleanser'] += 0.2
    
    for product in products:
        base_score = 0
        
        # SKIN TYPE MATCHING (with nuanced matching for combination and sensitive skin)
        # Check if product is suitable for the skin type
        if 'all' in product.skin_type:
            base_score += 2.5
        elif skin_type in product.skin_type:
            base_score += 3.5
        elif skin_type in combination_types:
            # For combination skin, products for oily or dry can work for specific areas
            if any(type in product.skin_type for type in oily_types + dry_types):
                base_score += 1.5
        elif skin_type in sensitive_types:
            # For sensitive skin, normal products might be ok but not oily/acne focused ones
            if any(type in product.skin_type for type in normal_types):
                base_score += 1.0
            # Penalize products that might be harsh for sensitive skin
            if 'acne-prone' in product.skin_type or 'oily' in product.skin_type:
                base_score -= 1.0
        elif skin_type in dry_types:
            # Dry skin benefits from products for mature skin sometimes
            if 'mature' in product.skin_type:
                base_score += 1.0
        elif skin_type in oily_types:
            # Oily skin often benefits from products for acne-prone skin
            if 'acne-prone' in product.skin_type:
                base_score += 1.0
        
        # CONCERN MATCHING (prioritize top concerns and give less weight to later ones)
        # Concerns are now sorted by confidence, with most important first
        for i, concern in enumerate(concerns):
            if concern in product.concerns:
                # Give more weight to top concerns
                weight = 1.0 if i == 0 else 0.8 if i == 1 else 0.6 if i == 2 else 0.4
                base_score += 2.0 * weight
        
        # ROUTINE BALANCE (create a complete skincare routine)
        # Apply category importance and prioritize a balanced routine
        category_score = category_importance.get(product.category, 0.5)
        
        # Limit duplicate categories (we don't want all serums, for example)
        if recommended_categories[product.category] >= 1:
            category_score *= 0.3  # Significantly reduce score for duplicate categories
        
        # SKIN TONE CONSIDERATIONS
        # Certain products work better for specific skin tones
        if product.category == 'suncare':
            if user_tone_group == 'light':
                base_score += 1.0  # Very important for fair skin
            else:
                base_score += 0.7  # Still important for everyone
        
        if 'brightening' in product.concerns or 'hyperpigmentation' in product.concerns:
            if user_tone_group == 'deep' or 'uneven tone' in concerns:
                base_score += 1.2
        
        # UNDERTONE MATCHING
        # Some products work better with specific undertones
        if 'warm' in product.name.lower() and user_undertone == 'warm':
            base_score += 0.5
        elif 'cool' in product.name.lower() and user_undertone == 'cool':
            base_score += 0.5
        
        # Calculate final score with small randomization for variety
        final_score = base_score + category_score + random.uniform(0, 0.3)
        
        scored_products.append((product, final_score, product.category))
    
    # Sort by score (descending)
    scored_products.sort(key=lambda x: x[1], reverse=True)
    
    # Build a balanced routine (select products from different categories)
    balanced_recommendations = []
    essential_categories = {'cleanser', 'moisturizer', 'suncare'}
    
    # First, include at least one of each essential category if possible
    for category in essential_categories:
        for product, score, product_category in scored_products:
            if product_category == category and product not in balanced_recommendations:
                balanced_recommendations.append(product)
                recommended_categories[category] += 1
                break
    
    # Then add other high-scoring products while maintaining category balance
    for product, score, category in scored_products:
        if product not in balanced_recommendations and len(balanced_recommendations) < max_recommendations:
            if recommended_categories[category] < 2:  # Limit to max 2 products per category
                balanced_recommendations.append(product)
                recommended_categories[category] += 1
    
    # If we don't have enough products yet, add the highest scoring remaining ones
    remaining_slots = max_recommendations - len(balanced_recommendations)
    if remaining_slots > 0:
        for product, score, category in scored_products:
            if product not in balanced_recommendations and len(balanced_recommendations) < max_recommendations:
                balanced_recommendations.append(product)
    
    # Log the recommended product categories for debugging
    logger.info(f"Recommended product categories: {dict(recommended_categories)}")
    
    return balanced_recommendations
