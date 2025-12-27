import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, LineString
import random
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import hashlib

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="RescueRoute | COMMAND OS", initial_sidebar_state="expanded")

# --- 1. THE AI BRAIN (PREDICTIVE MODELING) ---
@st.cache_resource
def train_ai_brain():
    np.random.seed(42)
    n_samples = 1500
    
    data = {
        'wind_speed_kmh': np.random.uniform(0, 250, n_samples),
        'rainfall_mm': np.random.uniform(0, 500, n_samples),   
        'seismic_mag': np.random.uniform(4, 9, n_samples),      
        'coastal_proximity': np.random.uniform(0, 1, n_samples), 
        'population_density': np.random.uniform(0, 10000, n_samples) 
    }
    df = pd.DataFrame(data)
    
    df['surge_radius'] = (df['rainfall_mm'] * 0.008) + (df['coastal_proximity'] * 3.0)
    df['wind_radius'] = (df['wind_speed_kmh'] - 60) * 0.05
    df['wind_radius'] = df['wind_radius'].clip(lower=0)
    df['gridlock_radius'] = (df['wind_speed_kmh'] * 0.01) + (df['population_density'] * 0.0001)
    
    model_surge = RandomForestRegressor(n_estimators=50).fit(df[['rainfall_mm', 'coastal_proximity']], df['surge_radius'])
    model_wind = RandomForestRegressor(n_estimators=50).fit(df[['wind_speed_kmh']], df['wind_radius'])
    model_gridlock = RandomForestRegressor(n_estimators=50).fit(df[['wind_speed_kmh', 'population_density']], df['gridlock_radius'])
    
    return model_surge, model_wind, model_gridlock

with st.spinner("Initializing Neural Link... Accessing IMD Satellite Feed..."):
    ai_surge, ai_wind, ai_gridlock = train_ai_brain()

# --- 2. GEOGRAPHY & ASSETS ---
CITY_DATABASE = {
    "🇮🇳 MUMBAI (South)": {
        "center": (18.9320, 72.8250), 
        "zoom": 13,
        "hq_virtual": {"name": "STATE DISASTER DEPOT (NDRF HQ)", "dist_km": 55, "eta_hours": 3.5},
        "hq_marker": (18.9800, 72.8400),
        "target_marker": (18.9000, 72.8150),
        "target_name": "Fishermen's Colony (South Colaba)",
        "assets": [
            {"name": "District Civil Hospital", "type": "MEDICAL", "coords": (18.9250, 72.8250), "tier": "Tier-1", "risk_profile": "HIGH"},
            {"name": "Coastal Police HQ", "type": "LOGISTICS", "coords": (18.9200, 72.8300), "tier": "Tier-2", "risk_profile": "HIGH"},
            {"name": "St. Xaviers University", "type": "STAGING", "coords": (18.9450, 72.8350), "tier": "Tier-1", "risk_profile": "LOW"},
            {"name": "Central Railway Warehouse", "type": "LOGISTICS", "coords": (18.9500, 72.8400), "tier": "Tier-2", "risk_profile": "LOW"},
            {"name": "Brabourne Stadium", "type": "SHELTER", "coords": (18.9322, 72.8275), "tier": "Tier-1", "risk_profile": "MED"}
        ]
    },
    "🇺🇸 ATLANTA (Metro)": {
        "center": (33.7756, -84.3963),
        "zoom": 13,
        "hq_virtual": {"name": "GEMA STATE OPS CENTER", "dist_km": 42, "eta_hours": 2.5},
        "hq_marker": (33.8200, -84.3800), 
        "target_marker": (33.7400, -84.4100),
        "target_name": "West End Community Center",
        "assets": [
            {"name": "Grady Memorial Hospital", "type": "MEDICAL", "coords": (33.7513, -84.3830), "tier": "Tier-1", "risk_profile": "HIGH"},
            {"name": "Bobby Dodd Stadium", "type": "STAGING", "coords": (33.7724, -84.3928), "tier": "Tier-1", "risk_profile": "LOW"},
            {"name": "Piedmont Park", "type": "SHELTER", "coords": (33.7838, -84.3755), "tier": "Tier-2", "risk_profile": "MED"},
            {"name": "Mercedes-Benz Stadium", "type": "LOGISTICS", "coords": (33.7550, -84.4000), "tier": "Tier-1", "risk_profile": "LOW"},
            {"name": "Centennial Park", "type": "SHELTER", "coords": (33.7603, -84.3935), "tier": "Tier-2", "risk_profile": "HIGH"}
        ]
    }
}

# --- 3. SESSION STATE ---
if 'timeline_phase' not in st.session_state: st.session_state.timeline_phase = 1 
if 'risk_map' not in st.session_state: st.session_state.risk_map = {"blue": None, "yellow": None, "red": None}
if 'sim_params' not in st.session_state: st.session_state.sim_params = {}
if 'staging_decision' not in st.session_state: st.session_state.staging_decision = None
if 'selected_city' not in st.session_state: st.session_state.selected_city = "🇮🇳 MUMBAI (South)"
if 'manual_blocks' not in st.session_state: st.session_state.manual_blocks = []
if 'protocol_omega' not in st.session_state: st.session_state.protocol_omega = False
if 'last_hazard_count' not in st.session_state: st.session_state.last_hazard_count = 0
if 'command_mode' not in st.session_state: st.session_state.command_mode = 'THREAT'
if 'custom_target' not in st.session_state: st.session_state.custom_target = None
if 'previous_mode' not in st.session_state: st.session_state.previous_mode = "☢️ THREAT INJECTOR"

# --- 4. DATA INGESTION ---
@st.cache_resource
def load_graph(city_name):
    active_city = CITY_DATABASE[city_name] 
    with st.spinner(f"Downloading Satellite Data for {city_name}..."):
        G = ox.graph_from_point(active_city["center"], dist=6000, network_type='drive')
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        return G

# --- 5. UTILITIES ---
def generate_zone(center, radius_km, noise_level=0.1):
    if radius_km <= 0.1: return None 
    points = []
    num_points = 30
    base_deg = radius_km / 111.0
    random.seed(42) 
    
    for i in range(num_points):
        angle = math.radians(float(i) / num_points * 360)
        noise = random.uniform(1.0 - noise_level, 1.0 + noise_level)
        r = base_deg * noise
        lat = center[0] + r * math.cos(angle)
        lon = center[1] + r * math.sin(angle)
        points.append((lon, lat))
    return Polygon(points)

def generate_traffic_layer(G, center_coords, gridlock_radius_km, pop_density, disaster_intensity):
    """ Feature 2: Predictive Traffic Visualization """
    traffic_edges = []
    if gridlock_radius_km <= 0: return traffic_edges
    
    spillover_radius_km = gridlock_radius_km * 1.5
    spillover_radius_deg = spillover_radius_km / 111.0
    center_point = Point(center_coords[1], center_coords[0])

    norm_density = min(pop_density / 10000.0, 1.0)
    norm_intensity = min(disaster_intensity / 250.0, 1.0)

    for u, v, k, data in G.edges(keys=True, data=True):
        node_u = G.nodes[u]
        if abs(node_u['y'] - center_coords[0]) > spillover_radius_deg * 1.2: continue
        if abs(node_u['x'] - center_coords[1]) > spillover_radius_deg * 1.2: continue

        edge_geom = None
        if 'geometry' in data:
            edge_geom = data['geometry']
            mid_idx = len(edge_geom.coords) // 2
            mid_pt = Point(edge_geom.coords[mid_idx])
            dist_deg = center_point.distance(mid_pt)
        else:
            p1 = (node_u['x'], node_u['y'])
            p2 = (G.nodes[v]['x'], G.nodes[v]['y'])
            edge_geom = LineString([p1, p2])
            dist_deg = center_point.distance(Point(p1))
        
        dist_km = dist_deg * 111.0
        if dist_km > spillover_radius_km: continue

        proximity_factor = max(0, 1.0 - (dist_km / spillover_radius_km))
        
        highway = data.get('highway', '')
        if isinstance(highway, list): highway = highway[0]
        
        cap_weight = 0.5 
        if highway in ['motorway', 'trunk', 'primary', 'primary_link']: cap_weight = 1.0
        elif highway in ['secondary', 'tertiary']: cap_weight = 0.8
        
        congestion_score = (norm_density * 40) + (norm_intensity * 40 * proximity_factor) + (cap_weight * 20 * proximity_factor)
        
        color = "#32CD32" 
        weight = 3
        opacity = 0.5
        
        if dist_km <= gridlock_radius_km:
            if congestion_score > 60:
                color = "#FF0000"
                weight = 5
                opacity = 0.8
            elif congestion_score > 30:
                color = "#FF4500"
                weight = 4
                opacity = 0.7
        else:
            if congestion_score > 50:
                color = "#FFA500"
                weight = 4
                opacity = 0.6
            elif congestion_score > 20:
                color = "#FFD700"
                weight = 3
                opacity = 0.6

        traffic_edges.append({'coords': [(lat, lon) for lon, lat in edge_geom.coords], 'color': color, 'weight': weight, 'opacity': opacity})
            
    return traffic_edges

def calculate_confidence_score(G, route, center_coords, gridlock_radius_km, pop_density, disaster_intensity, blue_poly, red_poly):
    """
    Feature 4: Data-Driven Confidence Scoring
    Calculates score based on:
    1. Traffic Chaos (Congestion on path)
    2. Distance Decay
    3. Hazard Proximity
    """
    if not route or len(route) < 2: return 0, "No Path"
    
    total_congestion = 0
    segments = 0
    center_point = Point(center_coords[1], center_coords[0])
    
    # 1. Traffic Chaos Factor
    norm_density = min(pop_density / 10000.0, 1.0)
    norm_intensity = min(disaster_intensity / 250.0, 1.0)
    spillover_radius_km = gridlock_radius_km * 1.5
    
    hazard_penalty = 0
    
    for u, v in zip(route[:-1], route[1:]):
        # Get edge data
        data = G.get_edge_data(u, v)[0]
        node_u = G.nodes[u]
        
        # Calculate distance from disaster center
        p1 = Point(node_u['x'], node_u['y'])
        dist_deg = center_point.distance(p1)
        dist_km = dist_deg * 111.0
        
        # Congestion Math (Same as Traffic Layer)
        proximity_factor = max(0, 1.0 - (dist_km / spillover_radius_km))
        
        highway = data.get('highway', '')
        if isinstance(highway, list): highway = highway[0]
        cap_weight = 0.5 
        if highway in ['motorway', 'trunk', 'primary']: cap_weight = 1.0
        
        # Calculate raw congestion for this segment
        segment_congestion = (norm_density * 40) + (norm_intensity * 40 * proximity_factor) + (cap_weight * 20 * proximity_factor)
        total_congestion += segment_congestion
        segments += 1
        
        # 3. Hazard Proximity Check
        # Check if this specific point is suspiciously close to a polygon (even if safe)
        if blue_poly and blue_poly.distance(p1) < 0.005: # ~500m
            hazard_penalty = max(hazard_penalty, 20) # Near Flood
        if red_poly and red_poly.distance(p1) < 0.005:
            hazard_penalty = max(hazard_penalty, 15) # Near Gridlock

    avg_congestion = total_congestion / max(1, segments)
    
    # 2. Distance Decay
    total_dist_km = nx.path_weight(G, route, weight='length') / 1000.0
    dist_penalty = total_dist_km * 0.5 # -0.5% per km
    
    # Final Formula
    # Base 100 - (Chaos * 0.4) - Distance - Hazard Risk
    raw_score = 100 - (avg_congestion * 0.4) - dist_penalty - hazard_penalty
    final_score = max(10, min(99, int(raw_score))) # Clamp between 10 and 99
    
    reason = "Optimal Conditions"
    if hazard_penalty > 15: reason = "High Hazard Proximity"
    elif avg_congestion > 50: reason = "Heavy Congestion Expected"
    elif dist_penalty > 15: reason = "Long-Range Logistics Risk"
    
    return final_score, reason

def vet_assets(assets, blue_poly, yellow_poly, red_poly):
    vetted = []
    safe_count = 0
    for asset in assets:
        pt = Point(asset['coords'][1], asset['coords'][0])
        status, color, reason = "SECURE", "green", "Optimal Location"
        
        if blue_poly and blue_poly.contains(pt):
            status, color, reason = "CRITICAL RISK", "red", "Inside Flood Surge Zone"
        elif red_poly and red_poly.contains(pt):
            status, color, reason = "GRIDLOCKED", "purple", "Inside Panic Traffic Zone"
        elif yellow_poly and yellow_poly.contains(pt):
            status, color, reason = "HIGH RISK", "orange", "Wind Damage Risk"
        else:
            safe_count += 1
            
        asset.update({'status': status, 'color': color, 'reason': reason})
        vetted.append(asset)
    return vetted, safe_count

def get_routing_graph(G, blue_poly, red_poly, manual_blocks, risk_penalty=50):
    G_routing = G.copy()
    
    # 1. Environmental Hazards (Flood) - Always Impassable (Physics)
    if blue_poly:
        for u, v, k, data in G_routing.edges(keys=True, data=True):
            pt = Point(G_routing.nodes[u]['x'], G_routing.nodes[u]['y'])
            if blue_poly.contains(pt):
                G_routing[u][v][k]['length'] = 1e9 
                
    # 2. Chaos/Gridlock (Red) - Variable Penalty
    if red_poly:
        for u, v, k, data in G_routing.edges(keys=True, data=True):
            pt = Point(G_routing.nodes[u]['x'], G_routing.nodes[u]['y'])
            if red_poly.contains(pt):
                G_routing[u][v][k]['length'] *= risk_penalty 
    
    # 3. Dynamic Hazards (Manual Blocks) - Variable Penalty
    for lat, lng in manual_blocks:
        u, v, k = ox.nearest_edges(G, lng, lat)
        if G_routing.has_edge(u, v, k):
            G_routing[u][v][k]['length'] *= risk_penalty
            
    return G_routing

# --- 6. SIDEBAR: MISSION CONTROL ---
st.sidebar.title("🚨 COMMAND OS")
st.sidebar.caption("Operation Samudra Raksha")

# GLOBAL RESET
if st.sidebar.button("🔄 GLOBAL RESET"):
    st.session_state.timeline_phase = 1
    st.session_state.risk_map = {"blue": None, "yellow": None, "red": None}
    st.session_state.staging_decision = None
    st.session_state.manual_blocks = []
    st.session_state.protocol_omega = False
    st.session_state.sim_params = {}
    st.session_state.custom_target = None
    st.rerun()

if st.session_state.timeline_phase == 1:
    new_city = st.sidebar.selectbox("📍 Operation Theater", list(CITY_DATABASE.keys()), index=list(CITY_DATABASE.keys()).index(st.session_state.selected_city))
    if new_city != st.session_state.selected_city:
        st.session_state.selected_city = new_city
        st.session_state.risk_map = {"blue": None, "yellow": None, "red": None}
        st.session_state.sim_params = {}
        st.rerun()
else:
    st.sidebar.info(f"📍 Theater: {st.session_state.selected_city}")

ACTIVE_CITY = CITY_DATABASE[st.session_state.selected_city]
G = load_graph(st.session_state.selected_city)

# PHASE 1
st.sidebar.header("1. Situation Room (T-2 Hours)")
if st.session_state.timeline_phase == 1:
    in_wind = st.sidebar.slider("🌪️ Cyclone Wind Speed (km/h)", 0, 250, 0)
    in_rain = st.sidebar.slider("🌧️ Rain Intensity (mm/hr)", 0, 200, 0)
    in_pop = st.sidebar.slider("👥 Population Density", 0, 10000, 5000)
    
    input_surge = pd.DataFrame({'rainfall_mm': [in_rain], 'coastal_proximity': [1.0]})
    input_wind = pd.DataFrame({'wind_speed_kmh': [in_wind]})
    input_gridlock = pd.DataFrame({'wind_speed_kmh': [in_wind], 'population_density': [in_pop]})

    pred_surge = ai_surge.predict(input_surge)[0] 
    pred_wind = ai_wind.predict(input_wind)[0]
    pred_gridlock_radius = ai_gridlock.predict(input_gridlock)[0]
    
    if in_wind < 40 and in_rain < 10:
        pred_surge, pred_wind, pred_gridlock_radius = 0, 0, 0
        st.sidebar.success("✅ SENSORS CLEAR")
    else:
        st.sidebar.warning("⚠️ THREAT DETECTED")
    
    if st.sidebar.button("GENERATE RISK MAP"):
        blue = generate_zone(ACTIVE_CITY["center"], pred_surge, 0.2)
        yellow = generate_zone(ACTIVE_CITY["center"], pred_wind, 0.05)
        red = generate_zone(ACTIVE_CITY["center"], pred_gridlock_radius, 0.1)
        st.session_state.risk_map = {"blue": blue, "yellow": yellow, "red": red}
        st.session_state.sim_params = {"radius_km": pred_gridlock_radius, "density": in_pop, "intensity": in_wind}
        st.session_state.timeline_phase = 2 
        st.rerun()

# PHASE 2 (Pre-positioning)
elif st.session_state.timeline_phase == 2:
    st.sidebar.success("✅ Risk Map Generated")
    st.sidebar.markdown("---")
    st.sidebar.header("2. The Golden Hour (T-1 Hour)")
    
    vetted_assets, safe_count = vet_assets(ACTIVE_CITY["assets"], st.session_state.risk_map["blue"], st.session_state.risk_map["yellow"], st.session_state.risk_map["red"])
    
    st.sidebar.subheader("♟️ Strategic Intelligence")
    
    if safe_count == 0:
        st.sidebar.error("🚨 CRITICAL: ALL LOCAL ASSETS COMPROMISED")
        st.sidebar.warning("⚠️ PROTOCOL OMEGA ACTIVATED")
        if st.sidebar.button("🚀 AUTHORIZE PROTOCOL OMEGA"):
            st.session_state.staging_decision = {"name": ACTIVE_CITY['hq_virtual']['name'], "coords": ACTIVE_CITY["hq_marker"], "type": "VIRTUAL_HQ"}
            st.session_state.protocol_omega = True
            st.rerun()
    else:
        st.sidebar.info(f"{safe_count} Safe Staging Areas Identified")
        st.sidebar.caption("AI Vetting Report:")
        for asset in vetted_assets:
            if asset['status'] == "SECURE":
                if st.sidebar.button(f"✅ Deploy to: {asset['name']}"):
                    st.session_state.staging_decision = asset
                    st.rerun()
            elif asset['status'] == "CRITICAL RISK":
                st.sidebar.error(f"❌ {asset['name']} (Flooded)")
            elif asset['status'] == "GRIDLOCKED":
                st.sidebar.warning(f"⚠️ {asset['name']} (Gridlock)")

    if st.session_state.staging_decision:
        if st.session_state.protocol_omega:
             st.sidebar.warning(f"⚠️ Rerouting to: {st.session_state.staging_decision['name']}")
        else:
             st.sidebar.success(f"Confirmed: {st.session_state.staging_decision['name']}")
             st.sidebar.info("✅ SECURE INGRESS ROUTE ESTABLISHED")
             
        if st.sidebar.button("🔴 TRIGGER LANDFALL"):
            st.session_state.timeline_phase = 3
            st.rerun()

# PHASE 3 (Reactive Response & Blue Corridor)
elif st.session_state.timeline_phase == 3:
    if st.session_state.protocol_omega:
        st.sidebar.error("⚠️ PROTOCOL OMEGA ACTIVE")
        st.sidebar.caption("Local infrastructure collapse verified. Command transferred to State Ops.")
    else:
        st.sidebar.error("⚡ LANDFALL CONFIRMED (T+0)")
        
    st.sidebar.markdown("---")
    st.sidebar.header("3. Reactive Response")
    st.sidebar.markdown(f"**🎯 TARGET:** {ACTIVE_CITY['target_name']}")
    
    # PHASE 3.1: Command Mode Toggle
    st.sidebar.subheader("🎮 Ops Mode")
    mode = st.sidebar.radio("Map Interaction:", ["☢️ THREAT INJECTOR", "🎯 MISSION PLANNING"])
    
    if mode != st.session_state.previous_mode:
        st.toast(f"Switched to {mode}", icon="🎮")
        st.session_state.previous_mode = mode
    
    if mode == "☢️ THREAT INJECTOR":
        st.session_state.command_mode = 'THREAT'
        st.sidebar.info("Click MAP to spawn hazards (Fire/Collapse).")
        if st.session_state.manual_blocks:
            st.sidebar.warning(f"{len(st.session_state.manual_blocks)} Hazards Active")
            if st.sidebar.button("Undo Last Hazard"):
                st.session_state.manual_blocks.pop()
                st.rerun()
    else:
        st.session_state.command_mode = 'MISSION'
        st.sidebar.info("Click MAP to set new Rescue Target.")
        if st.session_state.custom_target:
            st.sidebar.success("Target Locked: Custom Coords")
            if st.sidebar.button("Reset Target"):
                st.session_state.custom_target = None
                st.rerun()

# --- 7. MAIN DISPLAY ---
if st.session_state.protocol_omega and st.session_state.timeline_phase == 3:
    st.error("CRITICAL FAILURE: LOCAL INFRASTRUCTURE COLLAPSED. INITIATING PROTOCOL OMEGA.")
    st.title(f"Command Center: STATE OPS -> {st.session_state.selected_city}")
else:
    st.title(f"Command Center: {st.session_state.selected_city}")

# ROUTING LOGIC
route_local, route_blue_corridor = None, None
confidence_score, confidence_reason = 0, "N/A"

if st.session_state.timeline_phase == 3 and st.session_state.staging_decision:
    G_safe = get_routing_graph(G, st.session_state.risk_map["blue"], st.session_state.risk_map["red"], st.session_state.manual_blocks, risk_penalty=100)
    
    if st.session_state.custom_target:
        target_lat, target_lng = st.session_state.custom_target
        target_node = ox.nearest_nodes(G, target_lng, target_lat)
    else:
        target_node = ox.nearest_nodes(G, ACTIVE_CITY["target_marker"][1], ACTIVE_CITY["target_marker"][0])

    hq_node = ox.nearest_nodes(G, ACTIVE_CITY["hq_marker"][1], ACTIVE_CITY["hq_marker"][0])
    staging_coords = st.session_state.staging_decision['coords']
    staging_node = ox.nearest_nodes(G, staging_coords[1], staging_coords[0])

    # SCENARIO 1: PROTOCOL OMEGA
    if st.session_state.protocol_omega:
        try: 
            route_blue_corridor = nx.shortest_path(G_safe, hq_node, target_node, weight='length')
            # Calculate Confidence for the Omega Route
            confidence_score, confidence_reason = calculate_confidence_score(
                G, route_blue_corridor, ACTIVE_CITY["center"], 
                st.session_state.sim_params["radius_km"], 
                st.session_state.sim_params["density"], 
                st.session_state.sim_params["intensity"],
                st.session_state.risk_map["blue"], 
                st.session_state.risk_map["red"]
            )
        except: pass

    # SCENARIO 2: NORMAL OPS
    else:
        try: route_blue_corridor = nx.shortest_path(G_safe, hq_node, staging_node, weight='length')
        except: pass
        try: 
            route_local = nx.shortest_path(G_safe, staging_node, target_node, weight='length')
            # Calculate Confidence for the Local Route
            confidence_score, confidence_reason = calculate_confidence_score(
                G, route_local, ACTIVE_CITY["center"], 
                st.session_state.sim_params["radius_km"], 
                st.session_state.sim_params["density"], 
                st.session_state.sim_params["intensity"],
                st.session_state.risk_map["blue"], 
                st.session_state.risk_map["red"]
            )
        except: 
            pass

    # Check manual blocks for toast
    if len(st.session_state.manual_blocks) > st.session_state.last_hazard_count:
        st.toast("⚠️ Hazard Detected! Recalculating Logistics Chain...", icon="🔄")
        st.session_state.last_hazard_count = len(st.session_state.manual_blocks)

# METRICS
c1, c2, c3 = st.columns(3)
if st.session_state.timeline_phase == 1:
    c1.metric("Status", "MONITORING", delta="T-Minus 2 Hours")
    c2.metric("Threat Level", "ELEVATED", delta_color="inverse")
    c3.metric("Golden Hour", "OPEN", delta="Act Now")
elif st.session_state.timeline_phase == 2:
    c1.metric("Status", "PRE-POSITIONING", delta="T-Minus 1 Hour", delta_color="off")
    if st.session_state.staging_decision:
        status = "PROTOCOL OMEGA" if st.session_state.protocol_omega else "SECURE"
        c2.metric("Unit Location", "STAGING AREA", delta=status)
        c3.metric("Readiness", "100%", delta="Golden Hour Secured")
    else:
        c2.metric("Unit Location", "DISTANT HQ", delta="-55 km", delta_color="inverse")
        c3.metric("Readiness", "CRITICAL", delta="Deployment Required", delta_color="inverse")
elif st.session_state.timeline_phase == 3:
    if st.session_state.protocol_omega:
         c1.metric("Status", "TOTAL FAILURE", delta="Fail-Safe Active", delta_color="inverse")
    else:
         c1.metric("Status", "REACTIVE RESPONSE", delta="T+1 Hour", delta_color="inverse")
    
    if route_blue_corridor and st.session_state.protocol_omega:
        dist_display_val = int(nx.path_weight(G, route_blue_corridor, weight='length') / 1000)
        c2.metric("Rescue Route", f"{dist_display_val} km", delta="LONG-RANGE")
        
        # Phase 4 Metric: Confidence Score
        delta_color = "normal" if confidence_score > 80 else "off" if confidence_score > 50 else "inverse"
        c3.metric("AI Confidence", f"{confidence_score}%", delta=confidence_reason, delta_color=delta_color)
        
    elif route_local:
        safe_km = int(nx.path_weight(G, route_local, weight='length') / 1000)
        c2.metric("Rescue Route", f"{safe_km} km", delta="OPTIMAL")
        
        # Phase 4 Metric: Confidence Score
        delta_color = "normal" if confidence_score > 80 else "off" if confidence_score > 50 else "inverse"
        c3.metric("AI Confidence", f"{confidence_score}%", delta=confidence_reason, delta_color=delta_color)
        
    else:
        c2.metric("Rescue Route", "BLOCKED", delta="No Viable Path", delta_color="inverse")
        c3.metric("Ingress Security", "CRITICAL", delta="Path Failed", delta_color="inverse")

# MAP RENDERING
m = folium.Map(location=ACTIVE_CITY["center"], zoom_start=ACTIVE_CITY["zoom"], tiles="Cartodb Positron")

if st.session_state.risk_map["red"] and st.session_state.sim_params:
    params = st.session_state.sim_params
    traffic_lines = generate_traffic_layer(G, ACTIVE_CITY["center"], params["radius_km"], params["density"], params["intensity"])
    for road in traffic_lines:
        folium.PolyLine(road['coords'], color=road['color'], weight=road['weight'], opacity=road['opacity']).add_to(m)

for zone, color, tooltip in [("yellow", "#FFD700", "Wind"), ("blue", "#00BFFF", "Flood")]:
    if st.session_state.risk_map[zone]:
        folium.GeoJson(st.session_state.risk_map[zone], 
                       style_function=lambda x, c=color: {'fillColor': c, 'color': 'none', 'fillOpacity': 0.3}, 
                       tooltip=tooltip,
                       interactive=False).add_to(m)

folium.Marker(ACTIVE_CITY["hq_marker"], icon=folium.Icon(color="gray", icon="home"), tooltip="External HQ (Start)").add_to(m)

if st.session_state.timeline_phase >= 2:
    vetted, _ = vet_assets(ACTIVE_CITY["assets"], st.session_state.risk_map["blue"], st.session_state.risk_map["yellow"], st.session_state.risk_map["red"])
    for asset in vetted:
        color = "blue" if st.session_state.staging_decision and st.session_state.staging_decision['name'] == asset['name'] else asset['color']
        if color == "purple": color = "darkpurple"
        folium.Marker(asset['coords'], icon=folium.Icon(color=color, icon="info-sign"), tooltip=f"{asset['name']} ({asset['status']})").add_to(m)

if st.session_state.timeline_phase == 2 and st.session_state.staging_decision:
    G_deploy = get_routing_graph(G, st.session_state.risk_map["blue"], st.session_state.risk_map["red"], [], risk_penalty=100) 
    hq_node = ox.nearest_nodes(G, ACTIVE_CITY["hq_marker"][1], ACTIVE_CITY["hq_marker"][0])
    staging_coords = st.session_state.staging_decision['coords']
    staging_node = ox.nearest_nodes(G, staging_coords[1], staging_coords[0])
    try:
        route_ingress = nx.shortest_path(G_deploy, hq_node, staging_node, weight='length')
        coords = [ACTIVE_CITY["hq_marker"]]
        for u, v in zip(route_ingress[:-1], route_ingress[1:]):
            data = G[u][v][0]
            if 'geometry' in data: 
                line_coords = [(lat, lon) for lon, lat in data['geometry'].coords]
                coords.extend(line_coords)
            else: 
                coords.extend([(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])])
        folium.PolyLine(coords, color="#00f0ff", weight=6, opacity=0.9, tooltip="SECURE INGRESS ROUTE", dash_array="1, 10").add_to(m)
        folium.PolyLine(coords, color="#00f0ff", weight=12, opacity=0.3).add_to(m)
        m.fit_bounds(coords)
    except:
        line = [ACTIVE_CITY["hq_marker"], st.session_state.staging_decision['coords']]
        folium.PolyLine(line, color="red", weight=4, dash_array="5, 5", tooltip="NO VALID INGRESS ROUTE").add_to(m)
    folium.Marker(st.session_state.staging_decision['coords'], icon=folium.Icon(color="green", icon="play"), tooltip="UNIT DEPLOYED").add_to(m)

if st.session_state.timeline_phase == 3:
    if st.session_state.custom_target:
        folium.Marker(st.session_state.custom_target, icon=folium.Icon(color="red", icon="flag"), tooltip="DYNAMIC TARGET").add_to(m)
    else:
        folium.Marker(ACTIVE_CITY["target_marker"], icon=folium.Icon(color="red", icon="flag"), tooltip="TARGET").add_to(m)
    
    for lat, lng in st.session_state.manual_blocks:
        folium.Marker([lat, lng], icon=folium.Icon(color="black", icon="fire", prefix="fa"), tooltip="Hazard (Blocking Path)").add_to(m)
        
    all_points = []
    if route_blue_corridor:
        coords = [ACTIVE_CITY["hq_marker"]]
        for u, v in zip(route_blue_corridor[:-1], route_blue_corridor[1:]):
            data = G[u][v][0]
            if 'geometry' in data: 
                line_coords = [(lat, lon) for lon, lat in data['geometry'].coords]
                coords.extend(line_coords)
            else: 
                coords.extend([(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])])
        all_points.extend(coords)
        tooltip_txt = "BLUE CORRIDOR (Protocol Omega)" if st.session_state.protocol_omega else "SECURE INGRESS ROUTE"
        folium.PolyLine(coords, color="#00f0ff", weight=6, opacity=0.9, tooltip=tooltip_txt, dash_array="1, 10").add_to(m)
        folium.PolyLine(coords, color="#00f0ff", weight=12, opacity=0.3).add_to(m)
        
    if route_local and not st.session_state.protocol_omega:
        coords = [st.session_state.staging_decision['coords']]
        for u, v in zip(route_local[:-1], route_local[1:]):
            data = G[u][v][0]
            if 'geometry' in data: 
                line_coords = [(lat, lon) for lon, lat in data['geometry'].coords]
                coords.extend(line_coords)
            else: 
                coords.extend([(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])])
        all_points.extend(coords)
        folium.PolyLine(coords, color="#00FF00", weight=6, opacity=0.9, tooltip="Local Rescue Route").add_to(m)
        
    if all_points:
        m.fit_bounds(all_points)

map_output = st_folium(m, width=1200, height=600, returned_objects=["last_object_clicked_tooltip", "last_clicked"])

if st.session_state.timeline_phase == 2 and map_output['last_object_clicked_tooltip']:
    clicked_name = map_output['last_object_clicked_tooltip'].split(" (")[0]
    for asset in ACTIVE_CITY["assets"]:
        if asset['name'] == clicked_name:
            if asset['color'] in ["red", "purple"]: st.toast("🚫 Unsafe for Staging!")
            else: 
                st.session_state.staging_decision = asset
                st.rerun()

elif st.session_state.timeline_phase == 3 and map_output['last_clicked']:
    lat, lng = map_output['last_clicked']['lat'], map_output['last_clicked']['lng']
    if st.session_state.command_mode == 'THREAT':
        st.session_state.manual_blocks.append((lat, lng))
        st.toast("🔥 Hazard Injected", icon="🔥")
        st.rerun()
    elif st.session_state.command_mode == 'MISSION':
        try:
            target_node = ox.nearest_nodes(G, lng, lat)
            node_data = G.nodes[target_node]
            node_pt = Point(node_data['x'], node_data['y'])
            click_pt = Point(lng, lat)
            dist_deg = node_pt.distance(click_pt)
            dist_km = dist_deg * 111.0 
            if dist_km > 2.0:
                st.toast("🚫 Target Unreachable: Too far from road network", icon="🌊")
            else:
                st.session_state.custom_target = (lat, lng)
                st.toast("🎯 New Mission Target Set", icon="🎯")
                st.rerun()
        except Exception as e:
             st.toast("🚫 Error resolving target location", icon="⚠️")

st.info("ℹ️ **MISSION:** 1. Generate Risk Map 2. Deploy via Sidebar (or Authorize Omega) 3. Trigger Landfall")
