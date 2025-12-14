import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon, LineString
import random
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="RescueRoute | COMMAND OS", initial_sidebar_state="expanded")

# --- 1. GLOBAL CONFIGURATION & ASSET DATABASE ---
DEMO_CONFIG = {
    "🇺🇸 ATLANTA (HQ)": {
        "center": (33.7756, -84.3963),
        "start": (33.7812, -84.3734),
        "end": (33.7756, -84.4030),
        "zoom": 14,
        "assets": [
            {
                "name": "Bobby Dodd Stadium", "type": "STADIUM", "coords": (33.7724, -84.3928), 
                "cap": "High", "desc": "Heavy Airlift / Mass Shelter",
                "supplies": ["Heavy Lift Drones", "Mass Shelter Tents (500)", "MREs (10k units)"]
            },
            {
                "name": "Piedmont Park (South)", "type": "PARK", "coords": (33.7838, -84.3755), 
                "cap": "Massive", "desc": "Field Hospital / Triage",
                "supplies": ["Inflatable Triage Tents", "Water Tankers (5)", "Starlink Relay"]
            },
            {
                "name": "Emory Midtown Hosp.", "type": "MEDICAL", "coords": (33.7692, -84.3865), 
                "cap": "Critical", "desc": "Trauma Center / ICU",
                "supplies": ["L1 Trauma Kits", "O- Blood Supply", "Surgical Units"]
            },
            {
                "name": "Atlantic Station Deck", "type": "PARKING", "coords": (33.7922, -84.3967), 
                "cap": "High", "desc": "Vehicle Staging / Logistics",
                "supplies": ["Ambulance Fleet (20)", "Fuel Tankers", "Generators"]
            },
            {
                "name": "Centennial Olympic Park", "type": "PARK", "coords": (33.7603, -84.3935), 
                "cap": "Massive", "desc": "Command Post Alpha",
                "supplies": ["Mobile Command Unit", "Radio Repeaters", "Riot Gear"]
            },
            {
                "name": "Coca-Cola HQ Plaza", "type": "COMMERCIAL", "coords": (33.7709, -84.3965), 
                "cap": "Medium", "desc": "Civilian Evac Point",
                "supplies": ["Bus Fleet", "Water Bottles", "Blankets"]
            }
        ]
    },
    "🇮🇳 MUMBAI (Sector 9)": {
        "center": (18.9320, 72.8250),
        "start": (18.9220, 72.8347),
        "end": (18.9430, 72.8230),
        "zoom": 14,
        "assets": [
            {
                "name": "Wankhede Stadium", "type": "STADIUM", "coords": (18.9389, 72.8258), 
                "cap": "High", "desc": "Mass Casualty / Helipad",
                "supplies": ["Heavy Lift Helos", "Flood Lights", "MREs (20k units)"]
            },
            {
                "name": "Brabourne Stadium", "type": "STADIUM", "coords": (18.9322, 72.8275), 
                "cap": "High", "desc": "Secondary Shelter Zone",
                "supplies": ["Cots (2000)", "Hygiene Kits", "Water Purification"]
            },
            {
                "name": "Bombay Hospital", "type": "MEDICAL", "coords": (18.9416, 72.8291), 
                "cap": "Critical", "desc": "L1 Trauma / Surgery",
                "supplies": ["Antivenom", "Surgical Teams", "Oxygen Tanks"]
            },
            {
                "name": "Azad Maidan", "type": "PARK", "coords": (18.9360, 72.8310), 
                "cap": "Massive", "desc": "Riot Control / Triage",
                "supplies": ["Barricades", "Megaphones", "First Aid Posts"]
            },
            {
                "name": "Cross Maidan", "type": "PARK", "coords": (18.9330, 72.8280), 
                "cap": "High", "desc": "Logistics Hub",
                "supplies": ["Truck Fleet", "Sandbags", "Shovels"]
            },
            {
                "name": "Oval Maidan", "type": "PARK", "coords": (18.9300, 72.8280), 
                "cap": "Massive", "desc": "Civilian Assembly Area",
                "supplies": ["Evac Buses", "Food Packets", "Info Kiosks"]
            }
        ]
    }
}

# --- 2. SETUP ---
st.sidebar.markdown("## 🛡️ RESCUE ROUTE | AI")
st.sidebar.caption("v6.1")

selected_city_name = st.sidebar.selectbox(
    "📍 Select Operation Theater",
    list(DEMO_CONFIG.keys())
)
ACTIVE_CITY = DEMO_CONFIG[selected_city_name]

# --- 3. DATA INGESTION ---
@st.cache_resource
def load_graph(city_name):
    with st.spinner(f"Downloading Satellite Data for {city_name}..."):
        G = ox.graph_from_point(ACTIVE_CITY["center"], dist=2500, network_type='drive')
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        return G

G = load_graph(selected_city_name)

# --- 4. SESSION STATE ---
if 'last_city' not in st.session_state:
    st.session_state.last_city = selected_city_name
if st.session_state.last_city != selected_city_name:
    st.session_state.target_coords = None
    st.session_state.generated_polygon = None
    st.session_state.blocked_edges = [] 
    st.session_state.panic_edges = []   
    st.session_state.manual_incidents = []
    st.session_state.last_city = selected_city_name
    st.rerun()

if 'target_coords' not in st.session_state:
    st.session_state.target_coords = None
if 'generated_polygon' not in st.session_state:
    st.session_state.generated_polygon = None
if 'blocked_edges' not in st.session_state:
    st.session_state.blocked_edges = []
if 'panic_edges' not in st.session_state:
    st.session_state.panic_edges = []
if 'manual_incidents' not in st.session_state:
    st.session_state.manual_incidents = []

if 'last_threat_type' not in st.session_state:
    st.session_state.last_threat_type = "🌊 FLOOD" 
if 'last_intensity' not in st.session_state:
    st.session_state.last_intensity = 5

# --- 5. PHYSICS ENGINE (GENERATOR) ---
def generate_disaster_zone(center_lat, center_lon, type, intensity):
    points = []
    num_points = 20
    if type == "🌋 EARTHQUAKE":
        base_radius = (intensity - 3) * 0.008 
        for i in range(num_points):
            angle = math.radians(float(i) / num_points * 360)
            noise = random.uniform(0.7, 1.3) 
            r = base_radius * noise
            lat = center_lat + r * math.cos(angle)
            lon = center_lon + r * math.sin(angle)
            points.append((lon, lat))
    else:
        # FEATURE 1: SCIENTIFIC FLOOD METRIC
        base_radius = intensity * 0.0000025
        
        for i in range(num_points):
            angle = math.radians(float(i) / num_points * 360)
            noise = random.uniform(0.9, 1.1)
            r = base_radius * noise
            lat = center_lat + r * math.cos(angle)
            lon = center_lon + r * math.sin(angle)
            points.append((lon, lat))
    return Polygon(points)

# --- 6. LOGIC ENGINE (REACTION) ---
def detect_impact(G, hazard_poly, disaster_type):
    compromised = []
    panic = []
    
    panic_zone = hazard_poly.buffer(0.004) 
    
    for u, v, k, data in G.edges(keys=True, data=True):
        u_node = G.nodes[u]
        point_u = Point(u_node['x'], u_node['y'])
        
        in_hazard = hazard_poly.contains(point_u)
        in_panic_zone = panic_zone.contains(point_u)
        
        if in_hazard:
            if disaster_type == "🌊 FLOOD":
                compromised.append((u, v, k)) 
            else:
                if random.random() < 0.7: 
                    compromised.append((u, v, k))
                    
        elif in_panic_zone:
            highway_type = data.get('highway', '')
            if isinstance(highway_type, list): 
                highway_type = highway_type[0]
            
            major_roads = ['motorway', 'trunk', 'primary', 'secondary']
            
            if highway_type in major_roads:
                if random.random() < 0.8:
                    panic.append((u, v, k))
            else:
                if random.random() < 0.1:
                    panic.append((u, v, k))
                    
    return compromised, panic

def rank_staging_areas(hazard_poly, target_lat, target_lon, assets):
    safety_zone = hazard_poly.buffer(0.002) 
    target_point = Point(target_lon, target_lat)
    safe_assets = []
    
    for asset in assets:
        loc = Point(asset["coords"][1], asset["coords"][0])
        if not safety_zone.contains(loc):
            dist = loc.distance(target_point)
            asset["dist"] = dist
            safe_assets.append(asset)
            
    safe_assets.sort(key=lambda x: x["dist"])
    if not safe_assets: return {}

    selection = {}
    if len(safe_assets) > 0: selection["ALPHA"] = safe_assets[0]
    
    for asset in safe_assets:
        if asset["type"] in ["STADIUM", "PARK"] and asset != selection.get("ALPHA"):
            selection["BRAVO"] = asset
            break
            
    for asset in safe_assets:
        if asset["type"] == "MEDICAL" and asset != selection.get("ALPHA") and asset != selection.get("BRAVO"):
            selection["CHARLIE"] = asset
            break
            
    return selection

def calculate_confidence_score(route, G, hazard_poly, manual_blocks, ghost_route=None):
    if not route: return 0.0
    route_points = [Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in route]
    route_line = LineString(route_points)
    
    # --- 1. SAFETY CHECK ---
    intersects = route_line.intersects(hazard_poly)
    dist_deg = hazard_poly.distance(route_line)
    
    # Manual Block Penalty
    manual_penalty = 0
    for lat, lng in manual_blocks:
        block_point = Point(lng, lat)
        if route_line.distance(block_point) < 0.0001: 
            manual_penalty = 50 
            
    if intersects:
        base_score = 35.5 
    else:
        risk_free_distance = 0.01
        if dist_deg >= risk_free_distance:
            base_score = 99.2
        else:
            score_boost = (dist_deg / risk_free_distance) * 49.0
            base_score = 50.0 + score_boost
    
    safety_score = base_score - manual_penalty

    # --- 2. EFFICIENCY CHECK (NEW) ---
    efficiency_penalty = 0
    if ghost_route:
        # Calculate lengths
        ai_len = nx.path_weight(G, route, weight='length')
        ghost_len = nx.path_weight(G, ghost_route, weight='length')
        
        if ghost_len > 0:
            ratio = ai_len / ghost_len
            # If AI route is > 20% longer, start penalizing
            if ratio > 1.2:
                # Cap penalty at 15 points
                efficiency_penalty = min(15, (ratio - 1.2) * 20)
    
    final_score = safety_score - efficiency_penalty
    return round(max(0, min(final_score, 99.4)), 1)

# --- 7. SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ Command Interface")

interaction_mode = st.sidebar.radio("Map Interaction Mode", ["🔴 THREAT GENERATOR", "⚠️ REPORT INCIDENT"])

if interaction_mode == "🔴 THREAT GENERATOR":
    st.sidebar.caption("Click map to simulate Disaster.")
    threat_type = st.sidebar.radio("Threat Model", ["🌊 FLOOD", "🌋 EARTHQUAKE"])
    if threat_type == "🌊 FLOOD":
        intensity = st.sidebar.slider("River Discharge (m³/s)", 500, 10000, 2500, step=500)
    else:
        intensity = st.sidebar.slider("Seismic Magnitude", 4.0, 9.0, 6.5)
else:
    st.sidebar.caption("Click map to report road blockage.")
    if st.sidebar.button("Undo Last Incident"):
        if st.session_state.manual_incidents:
            st.session_state.manual_incidents.pop()
            st.rerun()
    threat_type = st.session_state.last_threat_type
    intensity = st.session_state.last_intensity

# AUTO-REFRESH LOGIC
if st.session_state.target_coords and interaction_mode == "🔴 THREAT GENERATOR":
    if (threat_type != st.session_state.last_threat_type) or (intensity != st.session_state.last_intensity):
        st.session_state.last_threat_type = threat_type
        st.session_state.last_intensity = intensity
        
        lat, lng = st.session_state.target_coords
        poly = generate_disaster_zone(lat, lng, threat_type, intensity)
        st.session_state.generated_polygon = poly
        
        blocked, panic = detect_impact(G, poly, threat_type)
        st.session_state.blocked_edges = blocked
        st.session_state.panic_edges = panic
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("🟢 RESET SIMULATION"):
    st.session_state.target_coords = None
    st.session_state.generated_polygon = None
    st.session_state.blocked_edges = []
    st.session_state.panic_edges = []
    st.session_state.manual_incidents = []
    st.rerun()

# --- 8. ROUTING ENGINE ---
orig_node = ox.nearest_nodes(G, ACTIVE_CITY["start"][1], ACTIVE_CITY["start"][0])
dest_node = ox.nearest_nodes(G, ACTIVE_CITY["end"][1], ACTIVE_CITY["end"][0])

# --- ROUTE A: THE GHOST (Standard GPS) ---
try:
    route_ghost = nx.shortest_path(G, orig_node, dest_node, weight='length')
except nx.NetworkXNoPath:
    route_ghost = None

# --- ROUTE B: AI SECURE (The One True Path) ---
G_secure = G.copy()
for u, v, k in st.session_state.blocked_edges:
    if G_secure.has_edge(u, v, k):
        G_secure[u][v][k]['length'] = 1e9 # Effectively blocked
for u, v, k in st.session_state.panic_edges:
    if G_secure.has_edge(u, v, k):
        G_secure[u][v][k]['length'] = G_secure[u][v][k].get('length', 10) * 50 # High penalty
# Manual Blocks
for lat, lng in st.session_state.manual_incidents:
    block_node = ox.nearest_nodes(G, lng, lat)
    for u, v, k in G_secure.edges(block_node, keys=True):
        G_secure[u][v][k]['length'] = 1e9

try:
    route_ai = nx.shortest_path(G_secure, orig_node, dest_node, weight='length')
except nx.NetworkXNoPath:
    route_ai = None

# --- 9. VISUALIZATION & METRICS ---
st.title(f"Command Center: {selected_city_name}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("System Status", "ONLINE", delta="Local: 12ms")

if st.session_state.target_coords:
    m2.metric("Target Locked", "ACTIVE", delta="Tracking")
    total_impact = len(st.session_state.blocked_edges) + len(st.session_state.panic_edges) + len(st.session_state.manual_incidents)
    m3.metric("Infra. Impact", f"{total_impact} Segments", delta="CRITICAL", delta_color="inverse")
else:
    m2.metric("Targeting System", "STANDBY", delta_color="off")
    m3.metric("Infra. Impact", "0", delta_color="off")

# METRICS DISPLAY
if route_ai:
    # Calculate Real Distances
    dist_ai = nx.path_weight(G, route_ai, weight='length')
    eta_ai = int((dist_ai / 11.0) / 60)
    
    poly = st.session_state.generated_polygon
    if poly:
        # Pass ghost route for efficiency comparison
        confidence = calculate_confidence_score(route_ai, G, poly, st.session_state.manual_incidents, route_ghost)
        total_hazards = len(st.session_state.blocked_edges) + len(st.session_state.panic_edges) + len(st.session_state.manual_incidents)
        
        if route_ai == route_ghost:
            m4.metric("AI Confidence", f"{confidence}%", delta="Standard Route Safe", delta_color="normal")
            st.success(f"✅ **ROUTE CLEARED:** {eta_ai} mins | Standard Corridor is Secure")
        else:
            # Ghost comparison logic
            ghost_score = calculate_confidence_score(route_ghost, G, poly, st.session_state.manual_incidents, None)
            if ghost_score < 40:
                ghost_status = "CRITICAL FAIL"
                ghost_delta = "inverse"
            else:
                ghost_status = "HIGH RISK"
                ghost_delta = "off"
            
            m4.metric("AI Confidence", f"{confidence}%", delta=f"vs Std GPS: {ghost_status}", delta_color="normal")
            st.success(f"✅ **AI REROUTE ACTIVE:** {eta_ai} mins | Avoids {total_hazards} Hazards")
            st.warning(f"⚠️ **STANDARD GPS (GREY):** {ghost_status} | Route Compromised")
            
    else:
        m4.metric("AI Confidence", "100%", delta="Baseline")
        st.info(f"🚀 **OPTIMAL ROUTE:** {eta_ai} mins")

else:
    m4.metric("AI Confidence", "0%", delta="PATH LOST", delta_color="inverse")

m = folium.Map(location=ACTIVE_CITY["center"], zoom_start=ACTIVE_CITY["zoom"], tiles="Cartodb Positron")

# DRAW DISASTER
if st.session_state.generated_polygon:
    fill_color = "#0000FF" if threat_type == "🌊 FLOOD" else "#8B4513"
    folium.GeoJson(
        st.session_state.generated_polygon,
        style_function=lambda x: {'fillColor': fill_color, 'color': fill_color, 'weight': 2, 'fillOpacity': 0.4}
    ).add_to(m)
    
    staging = rank_staging_areas(st.session_state.generated_polygon, ACTIVE_CITY["end"][0], ACTIVE_CITY["end"][1], ACTIVE_CITY["assets"])
    
    if "ALPHA" in staging:
        a = staging["ALPHA"]
        supplies_str = "<br>• ".join(a['supplies'])
        folium.Marker(a["coords"], icon=folium.Icon(color="red", icon="bolt", prefix="fa"), popup=f"<div style='min-width: 300px;'><h5>⚡ ALPHA</h5><hr>{supplies_str}</div>").add_to(m)
    if "BRAVO" in staging:
        b = staging["BRAVO"]
        supplies_str = "<br>• ".join(b['supplies'])
        folium.Marker(b["coords"], icon=folium.Icon(color="orange", icon="helicopter", prefix="fa"), popup=f"<div style='min-width: 300px;'><h5>🚁 BRAVO</h5><hr>{supplies_str}</div>").add_to(m)
    if "CHARLIE" in staging:
        c = staging["CHARLIE"]
        supplies_str = "<br>• ".join(c['supplies'])
        folium.Marker(c["coords"], icon=folium.Icon(color="green", icon="user-md", prefix="fa"), popup=f"<div style='min-width: 300px;'><h5>🏥 CHARLIE</h5><hr>{supplies_str}</div>").add_to(m)
        
    folium.Marker(st.session_state.target_coords, icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(m)

else:
    for asset in ACTIVE_CITY["assets"]:
        folium.Marker(asset["coords"], icon=folium.Icon(color="gray", icon="info-sign"), popup=asset['name']).add_to(m)

# DRAW IMPACT EDGES
fail_color = "#0000cc" if threat_type == "🌊 FLOOD" else "#4a3c31"
for u, v, k in st.session_state.blocked_edges:
    if G.has_edge(u, v, k):
        coords = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
        folium.PolyLine(coords, color=fail_color, weight=3, opacity=0.8).add_to(m)

for u, v, k in st.session_state.panic_edges:
    if G.has_edge(u, v, k):
        coords = [[G.nodes[u]['y'], G.nodes[u]['x']], [G.nodes[v]['y'], G.nodes[v]['x']]]
        folium.PolyLine(coords, color="#FFA500", weight=4, opacity=0.7).add_to(m)

# MANUAL INCIDENTS
for lat, lng in st.session_state.manual_incidents:
    folium.Marker([lat, lng], icon=folium.Icon(color="black", icon="ban", prefix="fa"), tooltip="INCIDENT").add_to(m)

# DRAW ROUTE A (SECURE - GREEN)
if route_ai:
    route_coords = [ACTIVE_CITY["start"]]
    for i in range(len(route_ai) - 1):
        u = route_ai[i]
        v = route_ai[i+1]
        edge_data = G[u][v][0] 
        if 'geometry' in edge_data:
            geo_coords = [(lat, lon) for lon, lat in edge_data['geometry'].coords]
            route_coords.extend(geo_coords)
        else:
            route_coords.extend([(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])])
    route_coords.append(ACTIVE_CITY["end"])
    folium.PolyLine(route_coords, color="#00FF00", weight=6, opacity=0.9, tooltip="SECURE ROUTE").add_to(m)

# DRAW GHOST (GREY)
if route_ghost and st.session_state.generated_polygon and route_ghost != route_ai:
    ghost_coords = [ACTIVE_CITY["start"]]
    for i in range(len(route_ghost) - 1):
        u = route_ghost[i]
        v = route_ghost[i+1]
        edge_data = G[u][v][0] 
        if 'geometry' in edge_data:
            geo_coords = [(lat, lon) for lon, lat in edge_data['geometry'].coords]
            ghost_coords.extend(geo_coords)
        else:
            ghost_coords.extend([(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])])
    ghost_coords.append(ACTIVE_CITY["end"])
    folium.PolyLine(ghost_coords, color="#333333", weight=5, opacity=0.8, dash_array="10, 10", tooltip="STANDARD GPS").add_to(m)

folium.Marker(ACTIVE_CITY["start"], icon=folium.Icon(color="green", icon="play")).add_to(m)
folium.Marker(ACTIVE_CITY["end"], icon=folium.Icon(color="blue", icon="flag")).add_to(m)

map_output = st_folium(m, width=1000, height=500, returned_objects=["last_clicked"])

if map_output['last_clicked']:
    clicked_lat = map_output['last_clicked']['lat']
    clicked_lng = map_output['last_clicked']['lng']
    
    if interaction_mode == "🔴 THREAT GENERATOR":
        if st.session_state.target_coords != (clicked_lat, clicked_lng):
            st.session_state.target_coords = (clicked_lat, clicked_lng)
            poly = generate_disaster_zone(clicked_lat, clicked_lng, threat_type, intensity)
            st.session_state.generated_polygon = poly
            blocked, panic = detect_impact(G, poly, threat_type)
            st.session_state.blocked_edges = blocked
            st.session_state.panic_edges = panic
            st.rerun()
            
    elif interaction_mode == "⚠️ REPORT INCIDENT":
        new_point = (clicked_lat, clicked_lng)
        if new_point not in st.session_state.manual_incidents:
            st.session_state.manual_incidents.append(new_point)
            st.rerun()
