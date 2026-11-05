-- =============================================================================
-- Production Pipeline: GDELT 2.0 Global Knowledge Graph (GKG) Extraction
-- Target Database: Google Cloud BigQuery (bigquery-public-data.gdelt_fq.gkg)
-- Objective: Partition-pruned, cost-optimized extraction of geopolitical supply
--            disruption and energy chokepoint events during crisis windows.
-- =============================================================================

WITH filtered_gkg AS (
  SELECT
    -- Temporal attributes
    PARSE_DATE('%Y%m%d', CAST(SUBSTR(CAST(DATE AS STRING), 1, 8) AS STRING)) AS publish_date,
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS publish_timestamp,
    
    -- Document metadata
    GKGRECORDID AS record_id,
    DocumentIdentifier AS document_url,
    SourceCommonName AS source_domain,
    
    -- Content & Themes
    V2Themes AS themes,
    V2Locations AS locations,
    V2Persons AS persons,
    V2Organizations AS organizations,
    V2Tone AS raw_tone,
    
    -- Tone Decomposition: GDELT Tone score (field 1)
    CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS gdelt_tone_polarity

  FROM
    `bigquery-public-data.gdelt_fq.gkg`
  WHERE
    -- 1. Partition Pruning: Restrict scanning window (2022 Acute Conflict Window)
    _PARTITIONDATE BETWEEN '2022-01-01' AND '2022-06-30'
    AND DATE >= 20220101000000
    AND DATE <= 20220630235959
    
    -- 2. Language & Source Quality Filters
    AND DocumentIdentifier IS NOT NULL
    AND SourceCommonName IS NOT NULL
    
    -- 3. Targeted Theme Filtering: Energy infrastructure, military crisis, and bottlenecks
    AND (
      REGEXP_CONTAINS(V2Themes, r'(CRISISLEX_CRISISLEXREC|ENERGY_SUPPLY|PIPELINE|MILITARY|SANCTIONS|EMBARGO|CHOKEPOINT|BLOCKADE)')
    )
),

deduplicated_events AS (
  SELECT
    publish_date,
    publish_timestamp,
    record_id,
    document_url,
    source_domain,
    gdelt_tone_polarity,
    themes,
    
    -- Flagging specific supply chokepoints and energy assets
    CASE
      WHEN REGEXP_CONTAINS(LOWER(themes), r'(strait_of_hormuz|suez_canal|bab_el_mandeb|malacca|bosphorus)') THEN 'MARITIME_CHOKEPOINT'
      WHEN REGEXP_CONTAINS(LOWER(themes), r'(nord_stream|druzhba|keystone|transneft|tapr)') THEN 'PIPELINE_INFRASTRUCTURE'
      WHEN REGEXP_CONTAINS(LOWER(themes), r'(embargo|sanction|export_ban|price_cap)') THEN 'REGULATORY_RESTRICTION'
      ELSE 'GENERAL_ENERGY_CRISIS'
    END AS bottleneck_taxonomy,

    -- Window function to eliminate duplicate syndicate wire reports within the same trading day
    ROW_NUMBER() OVER(
      PARTITION BY publish_date, source_domain, SUBSTR(document_url, 1, 60)
      ORDER BY publish_timestamp ASC
    ) AS intra_day_rank

  FROM
    filtered_gkg
)

-- Final Analytical Extract
SELECT
  publish_date,
  publish_timestamp,
  record_id,
  source_domain,
  document_url,
  bottleneck_taxonomy,
  gdelt_tone_polarity
FROM
  deduplicated_events
WHERE
  intra_day_rank = 1
ORDER BY
  publish_timestamp ASC;
