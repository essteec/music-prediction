# Complete Column Specifications
**All 20 Columns - Validation Rules & ML Impact**

Generated: November 10, 2025

---

## Column-by-Column Breakdown

### 1. **id** - Track Identifier
- **Type**: String
- **Valid Range**: Any unique value
- **Null Allowed**: ❌ No
- **Unique**: ✅ Yes (MUST be unique)
- **ML Impact**: None (identifier only)
- **Target Variable**: No
- **Validation**: Check for duplicates

---

### 2. **name** - Track Name
- **Type**: String
- **Valid Range**: Any text
- **Null Allowed**: ❌ No
- **ML Impact**: None (metadata only)
- **Target Variable**: No
- **Validation**: Check for nulls

---

### 3. **album_name** - Album Name
- **Type**: String
- **Valid Range**: Any text
- **Null Allowed**: ✅ Yes (some tracks may not have albums)
- **ML Impact**: None
- **Target Variable**: No
- **Validation**: None required

---

### 4. **artists** - Artist Name(s)
- **Type**: String
- **Valid Range**: Any text
- **Null Allowed**: ❌ No
- **ML Impact**: None (metadata only)
- **Target Variable**: No
- **Validation**: Check for nulls

---

### 5. **danceability** 🎯 TARGET
- **Type**: Float
- **Valid Range**: 0.000 to 1.000
- **Null Allowed**: ❌ No (TARGET VARIABLE)
- **ML Impact**: ⭐ HIGH
- **Target Variable**: ✅ **YES - PRIMARY TARGET**
- **Validation**: 
  - Must be in [0.0, 1.0]
  - No nulls allowed
  - Check for values outside range
- **Description**: How suitable a track is for dancing

---

### 6. **energy** 🎯 TARGET
- **Type**: Float
- **Valid Range**: 0.000 to 1.000
- **Null Allowed**: ❌ No (TARGET VARIABLE)
- **ML Impact**: ⭐ HIGH
- **Target Variable**: ✅ **YES - PRIMARY TARGET**
- **Validation**: 
  - Must be in [0.0, 1.0]
  - No nulls allowed
  - Check for values outside range
- **Description**: Intensity and activity level

---

### 7. **key** - Musical Key
- **Type**: Integer
- **Valid Range**: 0 to 11 (12 musical keys)
  - 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
- **Null Allowed**: ❌ No
- **ML Impact**: Low (categorical feature)
- **Target Variable**: No
- **Validation**: Must be integer in [0, 11]

---

### 8. **loudness** - Loudness in dB
- **Type**: Float
- **Valid Range**: -60.0 to 0.0 dB
- **Typical Range**: -40.0 to -5.0 dB
- **Null Allowed**: ❌ No
- **ML Impact**: Medium (correlates with energy)
- **Target Variable**: No
- **Validation**: 
  - Must be in [-60, 0]
  - Warn if outside [-40, -5] (unusual but valid)

---

### 9. **mode** - Major/Minor
- **Type**: Integer
- **Valid Range**: 0 or 1
  - 0 = Minor
  - 1 = Major
- **Null Allowed**: ❌ No
- **ML Impact**: Low (tonality feature)
- **Target Variable**: No
- **Validation**: Must be 0 or 1

---

### 10. **speechiness** - Speech Content
- **Type**: Float
- **Valid Range**: 0.000 to 1.000
- **Null Allowed**: ❌ No
- **ML Impact**: Medium
- **Target Variable**: No
- **Validation**: Must be in [0.0, 1.0]
- **Description**: Presence of spoken words vs music

---

### 11. **acousticness** - Acoustic Quality
- **Type**: Float
- **Valid Range**: 0.000 to 1.000
- **Null Allowed**: ❌ No
- **ML Impact**: Medium
- **Target Variable**: No
- **Validation**: Must be in [0.0, 1.0]
- **Description**: Acoustic vs electronic

---

### 12. **instrumentalness** - Instrumental Content
- **Type**: Float
- **Valid Range**: 0.000 to 1.000
- **Null Allowed**: ❌ No
- **ML Impact**: Medium
- **Target Variable**: No
- **Validation**: Must be in [0.0, 1.0]
- **Description**: Predicts whether track has vocals

---

### 13. **liveness** - Live Performance
- **Type**: Float
- **Valid Range**: 0.000 to 1.000
- **Null Allowed**: ❌ No
- **ML Impact**: Low
- **Target Variable**: No
- **Validation**: Must be in [0.0, 1.0]
- **Description**: Live audience presence

---

### 14. **valence** 🎯 TARGET (PRIMARY)
- **Type**: Float
- **Valid Range**: 0.000 to 1.000
- **Null Allowed**: ❌❌ **CRITICAL - NEVER NULL**
- **ML Impact**: ⭐⭐ **VERY HIGH**
- **Target Variable**: ✅ **YES - PRIMARY TARGET**
- **Validation**: 
  - **MUST** be in [0.0, 1.0]
  - **MUST NEVER** be null
  - This is your main research target!
- **Description**: Musical positiveness (happy vs sad)

---

### 15. **tempo** - Tempo in BPM
- **Type**: Float
- **Valid Range**: 20.0 to 300.0 BPM
- **Typical Range**: 60.0 to 180.0 BPM
- **Null Allowed**: ❌ No
- **ML Impact**: Medium (correlates with energy/danceability)
- **Target Variable**: No
- **Validation**: 
  - Must be in [20, 300]
  - Warn if outside [60, 180] (unusual but valid)

---

### 16. **duration_ms** - Duration in Milliseconds
- **Type**: Integer
- **Valid Range**: 1,000 to 3,600,000 ms (1 sec to 1 hour)
- **Typical Range**: 30,000 to 600,000 ms (30 sec to 10 min)
- **Null Allowed**: ❌ No
- **ML Impact**: Low
- **Target Variable**: No
- **Validation**: 
  - Must be > 0
  - Warn if < 30,000 or > 600,000 (unusual but valid)

---

### 17. **lyrics** - Song Lyrics
- **Type**: String (text)
- **Valid Range**: Any text
- **Null Allowed**: ✅ Yes (instrumentals have no lyrics)
- **Empty Allowed**: ✅ Yes
- **ML Impact**: ⭐⭐ **VERY HIGH** (primary NLP feature)
- **Target Variable**: No
- **Validation**: 
  - Can be null/empty for instrumentals
  - Check encoding (UTF-8)
  - Will need text preprocessing later

---

### 18. **year** ⚠️ KNOWN ISSUE
- **Type**: Integer
- **Valid Range**: 1900 to 2025
- **Null Allowed**: ❌ No
- **ML Impact**: Medium (temporal trends)
- **Target Variable**: No
- **Known Issues**: ⚠️ **Some values = 0 (INVALID)**
- **Validation**: 
  - Must be in [1900, 2025]
  - **Check for year = 0** ⚠️
  - Check for negative values
  - Check for future dates (> 2025)
- **Action Required**: Define cleaning strategy for year = 0

---

### 19. **genre** ⚠️ KNOWN ISSUE
- **Type**: String (categorical)
- **Valid Range**: Valid genre from genre_mappings.csv
- **Null Allowed**: ❌ No
- **ML Impact**: ⭐⭐ **VERY HIGH** (key categorical feature)
- **Target Variable**: No
- **Known Issues**: ⚠️ **Some values = NaN (INVALID)**
- **Validation**: 
  - Must not be null/NaN
  - Must not be empty string
  - Should match genre_mappings.csv
  - **Check for NaN values** ⚠️
- **Action Required**: Define cleaning strategy for NaN genres

---

### 20. **popularity** 🎯 TARGET
- **Type**: Integer
- **Valid Range**: 0 to 100
- **Null Allowed**: ⚠️ Maybe (old/obscure songs may lack popularity)
- **ML Impact**: ⭐ HIGH
- **Target Variable**: ✅ **YES - SECONDARY TARGET**
- **Validation**: 
  - Must be in [0, 100] if present
  - Null acceptable for very old songs
  - Check for values outside range
- **Description**: Spotify popularity score

---

## Summary Tables

### Target Variables (4 total) 🎯

| Column | Type | Range | Null OK? | Priority | Notes |
|--------|------|-------|----------|----------|-------|
| **valence** | float | 0.0-1.0 | ❌ NEVER | PRIMARY | Main research target |
| **energy** | float | 0.0-1.0 | ❌ No | HIGH | Secondary target |
| **danceability** | float | 0.0-1.0 | ❌ No | HIGH | Secondary target |
| **popularity** | int | 0-100 | ⚠️ Maybe | MEDIUM | Can be null for old songs |

### High ML Impact Features (8 total)

| Column | Range | Issues | Action |
|--------|-------|--------|--------|
| danceability 🎯 | 0.0-1.0 | Check range | Validate |
| energy 🎯 | 0.0-1.0 | Check range | Validate |
| valence 🎯 | 0.0-1.0 | Check range, nulls | **CRITICAL** |
| popularity 🎯 | 0-100 | Check range | Validate |
| lyrics | Text | Encoding | Preprocess |
| genre | Categorical | **NaN values** ⚠️ | **FIX REQUIRED** |
| speechiness | 0.0-1.0 | Check range | Validate |
| acousticness | 0.0-1.0 | Check range | Validate |

### Known Data Quality Issues ⚠️

| Issue | Column | Count | Impact | Priority |
|-------|--------|-------|--------|----------|
| NaN values | genre | TBD | HIGH | 🔥 URGENT |
| Zero values | year | TBD | MEDIUM | 🔥 URGENT |
| Range violations | All numeric | TBD | MEDIUM | HIGH |
| Duplicates | id | TBD | MEDIUM | HIGH |
| Missing targets | valence, energy, danceability | TBD | CRITICAL | 🔥 URGENT |

---

## Validation Checklist

### Phase 1: Quick Checks
- [ ] Count total rows
- [ ] Count unique IDs
- [ ] Count duplicates
- [ ] Check all columns exist

### Phase 2: Missing Values
- [ ] Check nulls in each column
- [ ] **CRITICAL**: Ensure valence has 0 nulls
- [ ] **CRITICAL**: Ensure energy has 0 nulls
- [ ] **CRITICAL**: Ensure danceability has 0 nulls
- [ ] Check popularity nulls (acceptable if <20%)

### Phase 3: Range Violations
- [ ] Validate all [0, 1] normalized features
- [ ] Validate key [0, 11]
- [ ] Validate mode [0, 1]
- [ ] Validate loudness [-60, 0]
- [ ] Validate tempo [20, 300]
- [ ] Validate duration > 0
- [ ] Validate popularity [0, 100]

### Phase 4: Specific Issues
- [ ] **Count year = 0** ⚠️
- [ ] Count year < 1900
- [ ] Count year > 2025
- [ ] **Count genre NaN** ⚠️
- [ ] Count genre empty
- [ ] Count invalid genres

### Phase 5: Decision Points
- [ ] Decide how to handle year = 0
- [ ] Decide how to handle genre NaN
- [ ] Decide how to handle duplicates
- [ ] Decide how to handle range violations
- [ ] Document all decisions

---

## Quick Reference: What Needs Fixing

### 🔥 URGENT (Blocking ML Pipeline)
1. **genre NaN values** - Cannot train without genres
2. **year = 0 values** - Invalid data
3. **valence/energy/danceability nulls** - Target variables must be complete

### ⚠️ HIGH Priority
4. Range violations in any numeric column
5. Duplicate IDs

### ✅ Lower Priority
6. Lyrics encoding issues (handle during preprocessing)
7. popularity nulls (acceptable for old songs)
8. Unusual but valid values (outliers)

---

**Use this reference when**:
- Writing validation scripts
- Making cleaning decisions
- Documenting data issues
- Planning preprocessing steps

**See Also**:
- `ml/DATA_VALIDATION_ROADMAP.md` - Complete roadmap
- `ml/PHASE1_CHECKLIST.md` - Progress tracker
- `scripts/comprehensive_validation.py` - Validation script
