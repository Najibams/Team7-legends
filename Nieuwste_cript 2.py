import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import plotly.express as px
from geopy.distance import geodesic
import pydeck as pdk
import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import datetime

# Streamlit configuratie
st.set_page_config(layout="wide", page_title="Vliegtuiggeluid & Weerseffecten")

# Titel en introductie
st.title("📊 Vliegtuiggeluid Analyse & Weersomstandigheden")
st.markdown("""
Deze app onderzoekt de relatie tussen vliegtuiggeluid (gemeten bij Schiphol) en weersomstandigheden.
""")

# Sidebar voor filters
st.sidebar.header("Filters")

selected_date_range = st.sidebar.date_input(
    "Selecteer datum bereik",
    value=[datetime(2025, 3, 23), datetime(2025, 3, 26)],
    min_value=datetime(2025, 1, 1),
    max_value=datetime(2025, 12, 31)
)

# Functie om de haul category te bepalen op basis van ICAO types
def determine_haul_category(icao_type):
    short_haul = ['CRJ2', 'CRJ9', 'B737', 'A320', 'B738', 'B739']
    medium_haul = ['B772', 'B788', 'B77L', 'B38M', 'A20N']
    long_haul = ['B77W', 'B789', 'B773', 'B78X']

    if icao_type in short_haul:
        return 'Short'
    elif icao_type in medium_haul:
        return 'Medium'
    elif icao_type in long_haul:
        return 'Long'
    else:
        return None

# Cargo en passagiers data per ICAO type
cargo_passenger_mapping = {
    'CRJ2': {'cargo_kg': 2500, 'passengers': 50},
    'CRJ9': {'cargo_kg': 3500, 'passengers': 90},
    'B737': {'cargo_kg': 20000, 'passengers': 140},
    'A320': {'cargo_kg': 20000, 'passengers': 150},
    'B738': {'cargo_kg': 20000, 'passengers': 160},
    'B739': {'cargo_kg': 30000, 'passengers': 180},
    'B772': {'cargo_kg': 50000, 'passengers': 400},
    'B788': {'cargo_kg': 50000, 'passengers': 240},
    'B77L': {'cargo_kg': 60000, 'passengers': 400},
    'B38M': {'cargo_kg': 60000, 'passengers': 200},
    'A20N': {'cargo_kg': 25000, 'passengers': 180},
    'B77W': {'cargo_kg': 70000, 'passengers': 400},
    'B789': {'cargo_kg': 70000, 'passengers': 290},
    'B773': {'cargo_kg': 80000, 'passengers': 400},
    'B78X': {'cargo_kg': 80000, 'passengers': 300},
}

# API-gegevens ophalen
@st.cache_data
def get_flight_data(start_date, end_date):
    start_timestamp = int(pd.to_datetime(start_date).timestamp())
    end_timestamp = int(pd.to_datetime(end_date).timestamp())
    url = f'https://sensornet.nl/dataserver3/event/collection/nina_events/stream?conditions%5B0%5D%5B%5D=time&conditions%5B0%5D%5B%5D=%3E%3D&conditions%5B0%5D%5B%5D={start_timestamp}&conditions%5B1%5D%5B%5D=time&conditions%5B1%5D%5B%5D=%3C&conditions%5B1%5D%5B%5D={end_timestamp}&conditions%5B2%5D%5B%5D=label&conditions%5B2%5D%5B%5D=in&conditions%5B2%5D%5B2%5D%5B%5D=21&conditions%5B2%5D%5B2%5D%5B%5D=32&conditions%5B2%5D%5B2%5D%5B%5D=33&conditions%5B2%5D%5B2%5D%5B%5D=34&args%5B%5D=aalsmeer&args%5B%5D=schiphol&fields%5B%5D=time&fields%5B%5D=location_short&fields%5B%5D=location_long&fields%5B%5D=duration&fields%5B%5D=SEL&fields%5B%5D=SELd&fields%5B%5D=SELe&fields%5B%5D=SELn&fields%5B%5D=SELden&fields%5B%5D=SEL_dB&fields%5B%5D=lasmax_dB&fields%5B%5D=callsign&fields%5B%5D=type&fields%5B%5D=altitude&fields%5B%5D=distance&fields%5B%5D=winddirection&fields%5B%5D=windspeed&fields%5B%5D=label&fields%5B%5D=hex_s&fields%5B%5D=registration&fields%5B%5D=icao_type&fields%5B%5D=serial&fields%5B%5D=operator&fields%5B%5D=tags'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        colnames = pd.DataFrame(data['metadata'])
        df = pd.DataFrame(data['rows'])
        df.columns = colnames.headers
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['date'] = df['time'].dt.date

        # Voeg Haul_Category kolom toe op basis van ICAO type
        if 'icao_type' in df.columns:
            df['Haul_Category'] = df['icao_type'].apply(determine_haul_category)
            df['cargo_kg'] = df['icao_type'].map(lambda x: cargo_passenger_mapping.get(x, {}).get('cargo_kg', 0))
            df['passengers'] = df['icao_type'].map(lambda x: cargo_passenger_mapping.get(x, {}).get('passengers', 0))

            # Verwijder 'None' waarden
            df = df[df['Haul_Category'].notna()]

        return df

    except Exception as e:
        st.error(f"Fout bij ophalen API data: {e}")
        return pd.DataFrame()

# Weergegevens laden
@st.cache_data
def load_weather_data():
    try:
        weather_data = {
            'YYYYMMDD': [20250323, 20250324, 20250325, 20250326],
            'date': pd.to_datetime(['2025-03-23', '2025-03-24', '2025-03-25', '2025-03-26']).date,
            'FH': [50, 60, 70, 55],  # Windsnelheid in 0.1 m/s
            'DD': [180, 190, 200, 185],  # Windrichting
            'U': [70, 75, 80, 72],  # Relatieve vochtigheid
            'N': [5, 6, 4, 3],  # Bewolking
            'T': [100, 105, 95, 98]  # Temperatuur in 0.1 °C
        }
        df_csv = pd.DataFrame(weather_data)
        return df_csv

    except Exception as e:
        st.error(f"Fout bij laden weergegevens: {e}")
        return pd.DataFrame()

# Haversine formule voor afstandsberekening
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# CSV data laden voor het kaartje
@st.cache_data
def load_map_data():
    try:
        # Bestandslocaties
        flight_data_path = r"flights_today_master (1).csv"
        station_data_path = r"station_1_clean (1).csv"
        
        df = pd.read_csv(flight_data_path)
        df2 = pd.read_csv(station_data_path)

        # Basis data cleaning
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Altitude_feet'] = pd.to_numeric(df['Altitude_feet'].str.replace(',', ''), errors='coerce')
        df['Altitude_meters'] = df['Altitude_feet'] * 0.3048

        # Schiphol coördinaten
        schiphol_coords = (52.3086, 4.7681)

        # Bereken afstand tot Schiphol
        df['Distance'] = df.apply(lambda row: haversine(schiphol_coords[0], schiphol_coords[1],
                                 row['Latitude'], row['Longitude']), axis=1)

        # Filter voor kaart
        df_nl = df[df['Distance'] <= 10]
        return df_nl

    except Exception as e:
        st.error(f"Fout bij laden kaartdata: {e}")
        return pd.DataFrame()

# Data laden
flight_data = get_flight_data(selected_date_range[0], selected_date_range[1])
weather_data = load_weather_data()
map_data = load_map_data()

# Check of we voldoende data hebben
if flight_data.empty or weather_data.empty:
    st.warning("Niet genoeg data beschikbaar voor de geselecteerde periode.")
    st.stop()

# Data samenvoegen
df_merged = pd.merge(flight_data, weather_data, on='date', how='inner')

# Weergegevens verwerken (indien beschikbaar)
if 'windspeed' in df_merged.columns:
    df_merged['windspeed_clean'] = df_merged['windspeed'] / 10  # Geen gebruik in model

if 'T' in df_merged.columns:
    df_merged['temperature'] = df_merged['T'] / 10

if 'U' in df_merged.columns:
    df_merged['humidity'] = df_merged['U']

if 'N' in df_merged.columns:
    df_merged['cloud_cover'] = df_merged['N']


# Tabs voor secties
tab1, tab2, tab3, tab4 = st.tabs(["Data Overzicht", "Geluid vs. Weersomstandigheden", "Geluidsvoorspelling", "dB tegen Passagiers en Vracht"])

with tab1:
    st.header("🔍 Data Overzicht")
    st.write(f"Gegevens van {selected_date_range[0]} tot {selected_date_range[1]}")

    # Kaartje toevoegen
    if not map_data.empty:
        st.subheader("Vliegtuiggeluid Heatmap rond Schiphol")



# === Instellingen ===
st.title("🛬 Geluidsintensiteit Aalsmeerbaan Meetstation")
st.markdown("*Laatste update: 2025-03-28 01:09:03 UTC door Team 7*")

# === Data laden ===
df = pd.read_csv(r"C:\Users\nasro\Downloads\echt.juist 1.csv")
df = df.dropna(subset=["Latitude", "Longitude", "SEL_dB", "FlightNumber"])



# === Slider: aantal unieke vluchten kiezen ===
unique_flights = df["FlightNumber"].unique()
min_vluchten = 1
max_vluchten = len(unique_flights)

aantal_vluchten = st.slider(
    "Aantal vluchten weergeven op de kaart",
    min_value=min_vluchten,
    max_value=max_vluchten,
    value=min(50, max_vluchten),  # start bij 50 of minder
    step=1
)

# === dB Filter in sidebar ===
st.sidebar.header("🎚️ Filter op Geluidsniveau (SEL_dB)")

# Bereik bepalen uit data
min_db = int(df["SEL_dB"].min())
max_db = int(df["SEL_dB"].max())

sel_db_range = st.sidebar.slider(
    "Toon metingen binnen dit dB-bereik:",
    min_value=min_db,
    max_value=max_db,
    value=(min_db, max_db),
    step=1
)


ZONES = {
   "Zone 1 - Kern": {
       "radius": 0.5,  # 500 meter
       "fill_color": [255, 0, 0, 180],  # Rood
       "edge_color": [255, 0, 0, 255],
       "description": "0-500m vanaf meetstation",
       "zone_id": "kern"
   },
   "Zone 2 - Midden": {
       "radius": 1.0,
       "fill_color": [255, 140, 0, 160],  # Oranje
       "edge_color": [255, 140, 0, 255],
       "description": "500m-1km vanaf meetstation",
       "zone_id": "midden"
   },
   "Zone 3 - Buitenring": {
       "radius": 2.0,
       "fill_color": [255, 255, 0, 140],  # Geel
       "edge_color": [255, 255, 0, 255],
       "description": "1-2km vanaf meetstation",
       "zone_id": "buiten"
   }
}

from math import radians, cos, sin, asin, sqrt

# Coördinaten van Aalsmeerbaan
AALSMEERBAAN = {'lat': 52.273889, 'lon': 4.781944}
SCHIPHOL = {'lat': 52.3086, 'lon': 4.7681}


# Haversine functie om afstand tot Aalsmeerbaan te berekenen
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# Voeg afstand toe aan de DataFrame
df["distance_to_runway"] = df.apply(
    lambda row: haversine_distance(
        row["Latitude"], row["Longitude"],
        AALSMEERBAAN['lat'], AALSMEERBAAN['lon']
    ),
    axis=1
)

# Voeg zone toe aan de DataFrame
def get_zone(distance):
    if distance < 0.5:
        return "Zone 1 - Kern"
    elif distance < 1.0:
        return "Zone 2 - Midden"
    else:
        return "Zone 3 - Buitenring"

df["zone"] = df["distance_to_runway"].apply(get_zone)



# Filter data op geselecteerde vluchten
geselecteerde_vluchten = unique_flights[:aantal_vluchten]
filtered_df = df[df["FlightNumber"].isin(geselecteerde_vluchten)]

# === Data voorbereiden voor PathLayer ===
vluchten_pad_data = []
for FlightNumber, group in filtered_df.groupby("FlightNumber"):
    path = group.sort_values(by="timestamp") if "timestamp" in group else group
    path_coords = path[["Longitude", "Latitude"]].values.tolist()
    if len(path_coords) > 1:
        vluchten_pad_data.append({
            "FlightNumber": FlightNumber,
            "path": path_coords
        })

# === Pydeck Layers ===
path_layer = pdk.Layer(
    "PathLayer",
    data=vluchten_pad_data,
    get_path="path",
    get_color=[255, 255, 0],  # geel
    width_scale=20,
    width_min_pixels=1,
    opacity=0.4,
    pickable=True
)



from geopy.distance import geodesic

# Bereken vlucht-afstanden en categoriseer als haul type
def bereken_vluchtafstand(vlucht_df):
    punten = vlucht_df[["Latitude", "Longitude"]].dropna().values
    afstand = 0
    for i in range(len(punten) - 1):
        afstand += geodesic(punten[i], punten[i + 1]).km
    return afstand

haul_data = []

for flight_id, group in filtered_df.groupby("FlightNumber"):
    afstand = bereken_vluchtafstand(group)
    if afstand < 1500:
        haul = "Short Haul"
        color = [0, 255, 0]  # Groen
    elif afstand < 3500:
        haul = "Middle Haul"
        color = [255, 165, 0]  # Oranje
    else:
        haul = "Long Haul"
        color = [255, 0, 0]  # Rood

    group_sorted = group.sort_values(by="timestamp") if "timestamp" in group else group
    path_coords = group_sorted[["Longitude", "Latitude"]].values.tolist()

    if len(path_coords) > 1:
        haul_data.append({
            "FlightNumber": flight_id,
            "path": path_coords,
            "haul_type": haul,
            "color": color
        })









# === Pydeck Layers ===
haul_layer = pdk.Layer(
    "PathLayer",
    data=haul_data,
    get_path="path",
    get_color="color",  # 👈 Kleur per haul
    width_scale=20,
    width_min_pixels=1,
    opacity=0.5,
    pickable=True
)


# Meetstation markeerlaag
aalsmeerbaan_coords = [4.781944, 52.273889]
station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame([{"lon": aalsmeerbaan_coords[0], "lat": aalsmeerbaan_coords[1]}]),
    get_position=["lon", "lat"],
    get_radius=200,
    get_fill_color=[0, 0, 255, 200],
    pickable=True
)



# View instellingen
view_state = pdk.ViewState(
    longitude=aalsmeerbaan_coords[0],
    latitude=aalsmeerbaan_coords[1],
    zoom=1.6,
    pitch=30
)

# === Extra: ScatterLayer voor SEL_dB-waardes in de zones ===

def get_db_color(sel_db):
    """Bepaal kleur op basis van geluidsniveau"""
    if sel_db >= 70:
        return [255, 0, 0, 180]    # Rood
    elif sel_db >= 55:
        return [255, 165, 0, 160]  # Oranje
    else:
        return [255, 255, 0, 130]  # Geel

# Voeg kleur toe op basis van SEL_dB
filtered_df["color"] = filtered_df["SEL_dB"].apply(get_db_color)

# Voeg radius toe op basis van SEL_dB (schaalbaar)
filtered_df["radius"] = filtered_df["SEL_dB"] * 2.5  # pas factor aan voor grotere/kleinere bollen


# Alleen rijen met geldige SEL_dB
db_points = filtered_df[
    (filtered_df["SEL_dB"].notna()) &
    (filtered_df["SEL_dB"] >= sel_db_range[0]) &
    (filtered_df["SEL_dB"] <= sel_db_range[1])
].copy()


# Voeg kleur + radius toe op basis van SEL_dB
db_points["color"] = db_points["SEL_dB"].apply(get_db_color)
db_points["radius"] = db_points["SEL_dB"] * 2.5

# ScatterplotLayer alleen voor meetpunten met dB
decibel_punten_layer = pdk.Layer(
    "ScatterplotLayer",
    data=db_points,
    get_position=["Longitude", "Latitude"],
    get_fill_color="color",
    get_radius="radius",
    pickable=True
)

# === Cirkel tussen Schiphol en Aalsmeerbaan ===
# Gemiddeld punt (ongeveer midden van het aan/uitvlieggebied)
circle_center = {
    "lat": (AALSMEERBAAN["lat"] + SCHIPHOL["lat"]) / 2,
    "lon": (AALSMEERBAAN["lon"] + SCHIPHOL["lon"]) / 2
}

circle_layer = pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame([circle_center]),
    get_position=["lon", "lat"],
    get_radius=6000,  # in meters
    get_fill_color=[0, 255, 0, 80],  # Groen met transparantie
    stroke=True,
    get_line_color=[0, 200, 0],
    get_line_width=10,
    pickable=False
)




st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=view_state,
    layers=[
        path_layer,
        station_layer,
        haul_layer,
        decibel_punten_layer,
        circle_layer  # ✅ Toegevoegd
    ],
    tooltip={
    "html": """
<b>✈️ Vlucht:</b> {FlightNumber}<br>
<b>🎧 SEL_dB:</b> {SEL_dB}
""",
    "style": {
        "backgroundColor": "rgba(0,0,0,0.8)",
        "color": "white"
    }
}

))


# Interactieve analyse per zone
st.markdown("### 📊 Geluidsanalyse per Zone")

# Tabs voor Zonering & Statistieken
tab1, tab2 = st.tabs(["🗺️ Zonering", "📊 Statistieken"])

with tab1:
    for zone_name, zone_info in ZONES.items():
        zone_data = filtered_df[filtered_df['zone'] == zone_name]
        st.markdown(f"#### {zone_name}")
        cols = st.columns(3)
        with cols[0]:
            avg_zone = zone_data['SEL_dB'].mean()
            overall_avg = filtered_df['SEL_dB'].mean()
            delta = avg_zone - overall_avg if pd.notnull(avg_zone) else 0
            st.metric(
                "Gemiddeld dB",
                f"{avg_zone:.1f}" if pd.notnull(avg_zone) else "–",
                delta=f"{delta:.1f} t.o.v. gemiddeld" if pd.notnull(avg_zone) else ""
            )
        with cols[1]:
            max_val = zone_data['SEL_dB'].max()
            st.metric("Maximum dB", f"{max_val:.1f}" if pd.notnull(max_val) else "–")
        with cols[2]:
            st.metric("Aantal metingen", len(zone_data))
        st.markdown(f"*{zone_info['description']}*")

        if len(filtered_df) > 0:
            st.progress(len(zone_data) / len(filtered_df))
        else:
            st.progress(0)

with tab2:
    if 'timestamp' in filtered_df.columns:
        filtered_df['hour'] = pd.to_datetime(filtered_df['timestamp'], errors='coerce').dt.hour
        hourly_avg = filtered_df.groupby(['hour', 'zone'])['SEL_dB'].mean().unstack()
        st.line_chart(hourly_avg)

    stats_df = filtered_df.groupby('zone').agg({
        'SEL_dB': ['mean', 'min', 'max', 'count'],
        'FlightNumber': 'nunique' if 'FlightNumber' in filtered_df.columns else 'count'
    }).round(1)

    stats_df.columns = ['Gem. dB', 'Min dB', 'Max dB', 'Aantal metingen', 'Aantal vluchten']
    st.dataframe(stats_df.style.background_gradient(subset=['Gem. dB'], cmap='YlOrRd'))

# 🎨 Legenda
st.markdown("### 🎨 Legenda")
st.markdown("""
<style>
.gradient-box {
   height: 24px;
   border-radius: 4px;
   margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

for zone_name, zone_info in ZONES.items():
    color = zone_info["fill_color"]
    rgba = f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3] / 255})"
    st.markdown(f"""
<div style="display: flex; align-items: center; margin: 10px 0;">
<div class="gradient-box" style="width: 100px; background-color: {rgba};"></div>
<div style="margin-left: 10px;">
<strong>{zone_name}</strong><br>
<small>{zone_info['description']}</small>
</div>
</div>
    """, unsafe_allow_html=True)

# 📥 Downloadknop
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download geanalyseerde data",
    data=csv,
    file_name=f"geluidsdata_zones_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)


    # Nieuwe visualisaties voor Haul Category
if not flight_data.empty:
        st.subheader("Geluidsniveaus per Vliegtuigcategorie")
    
        # Filter op vliegtuigen met een hoogte van maximaal 500 meter
        filtered_flight_data_500m = flight_data[flight_data['altitude'] <= 500]  # 500 meter in feet
    
        # Controleer of de benodigde kolommen aanwezig zijn
        if 'Haul_Category' in filtered_flight_data_500m.columns and 'SEL_dB' in filtered_flight_data_500m.columns:
            # Gemiddeld en totaal SEL_dB per Haul Category
            avg_sel_db = filtered_flight_data_500m.groupby('Haul_Category')['SEL_dB'].mean().reset_index()
            total_sel_db = filtered_flight_data_500m.groupby('Haul_Category')['SEL_dB'].sum().reset_index()
    
            # Maak een figuur met twee subplots
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    
            # Gemiddeld SEL_dB
            sns.barplot(x='Haul_Category', y='SEL_dB', data=avg_sel_db, palette='viridis', ax=ax[0])
            ax[0].set_title('Gemiddeld SEL_dB per Vliegtuigcategorie bij max 500m hoogte', fontsize=14)
            ax[0].set_xlabel('Vliegtuigcategorie', fontsize=12)
            ax[0].set_ylabel('Gemiddeld SEL_dB', fontsize=12)
    
            # Totaal SEL_dB
            sns.barplot(x='Haul_Category', y='SEL_dB', data=total_sel_db, palette='plasma', ax=ax[1])
            ax[1].set_title('Totaal SEL_dB per Vliegtuigcategorie bij max 500m hoogte', fontsize=14)
            ax[1].set_xlabel('Vliegtuigcategorie', fontsize=12)
            ax[1].set_ylabel('Totaal SEL_dB', fontsize=12)
    
            # Toon de figuren
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("De benodigde kolommen ('Haul_Category' en 'SEL_dB') ontbreken in de data")

with tab2:
    st.header("📈 Geluid vs. Weersomstandigheden")
    
    plot_options = []
    
    if 'windspeed_clean' in df_merged.columns and 'lasmax_dB' in df_merged.columns:
        plot_options.append("Geluid vs. Windsnelheid")
    
    if 'temperature' in df_merged.columns and 'lasmax_dB' in df_merged.columns:
        plot_options.append("Geluid vs. Temperatuur")
    
    if 'humidity' in df_merged.columns and 'lasmax_dB' in df_merged.columns:
        plot_options.append("Geluid vs. Relatieve Vochtigheid")
    
    if 'cloud_cover' in df_merged.columns and 'lasmax_dB' in df_merged.columns:
        plot_options.append("Geluid vs. Bewolking")
    
    if 'altitude' in df_merged.columns and 'lasmax_dB' in df_merged.columns:
        plot_options.append("Geluid vs. Vlieghoogte")
    
    if 'type' in df_merged.columns and 'lasmax_dB' in df_merged.columns:
        plot_options.append("Geluid per vliegtuigtype")
    
    if 'time' in df_merged.columns and 'lasmax_dB' in df_merged.columns:
        plot_options.append("Geluid vs. Tijdstip van de dag")
    
    if not plot_options:
        st.warning("Geen geschikte data beschikbaar voor visualisaties.")
        st.stop()
    
    selected_plot = st.selectbox("Kies een visualisatie:", plot_options)
    
    # Functie voor plotten
    def create_plot(x, y, xlabel, ylabel, title, hue=None):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_merged, x=x, y=y, hue=hue, alpha=0.7, palette='viridis')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    # Plot weergeven op basis van selectie
    if selected_plot == "Geluid vs. Windsnelheid" and 'windspeed_clean' in df_merged.columns:
        create_plot('windspeed_clean', 'lasmax_dB', 'Windsnelheid (m/s)', 'Max geluidsniveau (dB)', 'Geluid vs. Windsnelheid')
    
    elif selected_plot == "Geluid vs. Temperatuur" and 'temperature' in df_merged.columns:
        create_plot('temperature', 'lasmax_dB', 'Temperatuur (°C)', 'Max geluidsniveau (dB)', 'Geluid vs. Temperatuur')
    
    elif selected_plot == "Geluid vs. Relatieve Vochtigheid" and 'humidity' in df_merged.columns:
        create_plot('humidity', 'lasmax_dB', 'Relatieve Vochtigheid (%)', 'Max geluidsniveau (dB)', 'Geluid vs. Relatieve Vochtigheid')
    
    elif selected_plot == "Geluid vs. Bewolking" and 'cloud_cover' in df_merged.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_merged, x='cloud_cover', y='lasmax_dB', ci=None)
        plt.xlabel('Bewolking (octanten)')
        plt.ylabel('Gemiddeld max geluidsniveau (dB)')
        plt.title('Gemiddeld geluidsniveau per bewolkingsgraad')
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()
    
    elif selected_plot == "Geluid vs. Vlieghoogte" and 'altitude' in df_merged.columns:
        # Filter data boven 500 meter
        filtered_data = df_merged[df_merged['altitude'] > 500]
        
        plt.figure(figsize=(10, 6))
        
        # Logarithmische transformatie van de hoogte en geluidsniveau
        filtered_data['log_altitude'] = np.log(filtered_data['altitude'])
        
        # Scatterplot met logaritmische waarden
        sns.scatterplot(data=filtered_data, x='log_altitude', y='SEL_dB', alpha=0.7, palette='viridis', label='Data')
        
        # Regressie met logaritmische hoogte
        sns.regplot(data=filtered_data, x='log_altitude', y='SEL_dB', scatter=False, color='red', label='Log. Regressielijn')
        
        plt.xlabel('Log(Hoogte (m))')
        plt.ylabel('Log(Max geluidsniveau (dB))')
        plt.title('Geluid vs. Vlieghoogte (alleen >500m)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

    elif selected_plot == "Geluid vs. Tijdstip van de dag" and 'time' in df_merged.columns:
        df_merged['hour'] = df_merged['time'].dt.hour
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_merged, x='hour', y='lasmax_dB', ci=None)
        plt.xlabel('Uur van de dag')
        plt.ylabel('Gemiddeld max geluidsniveau (dB)')
        plt.title('Gemiddeld geluidsniveau per uur van de dag')
        plt.ylim(df_merged['lasmax_dB'].min(), df_merged['lasmax_dB'].max())
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

with tab3:
    st.header("🔮 Geluidsvoorspellingen")

    # Dropdown voor Haul_Category
    selected_haul = st.selectbox("Selecteer Haul Category", ['Short', 'Medium', 'Long'])

    # Filter de flight_data op basis van de geselecteerde Haul_Category
    filtered_flight_data = flight_data[flight_data['Haul_Category'] == selected_haul]

    if filtered_flight_data.empty:
        st.warning("Geen data beschikbaar voor de geselecteerde Haul Category.")
        st.stop()

    # Train het model met de geselecteerde data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Zorg ervoor dat we alleen de benodigde kolommen gebruiken
    X = filtered_flight_data[['altitude', 'distance']]
    y = filtered_flight_data['SEL_dB']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model trainen
    model.fit(X_train, y_train)

    # Sliders voor invoer
    col1, col2 = st.columns(2)
    with col1:
        alt = st.slider("Hoogte (m)", 500, 5000, 2000)
    with col2:
        dist = st.slider("Afstand (m)", 100, 10000, 2000)

    # Voorspelling maken
    prediction = model.predict([[alt, dist]])[0]

    # Pas de voorspelling aan op basis van Haul_Category
    if selected_haul == "Short" and alt <= 1800:
        prediction = max(prediction, 39)  # Voorbeeld: voor short haul bij hoogte <= 1800, waarde onder 40 dB
    elif selected_haul == "Medium" and alt <= 2400:
        prediction = max(prediction, 39)  # Voorbeeld: voor medium haul bij hoogte <= 2400, waarde onder 40 dB
    elif selected_haul == "Long" and alt <= 3000:
        prediction = max(prediction, 39)  # Voorbeeld: voor long haul bij hoogte <= 3000, waarde onder 40 dB

    # Waarschuwing voor hoge geluidsniveaus
    if prediction > 71:
        st.warning("⚠️ Het voorspelde geluidsniveau is hoog en kan als storend worden ervaren!")

    st.metric("Voorspeld geluidsniveau", f"{prediction:.1f} dB")

with tab4:
    st.header("📊 Geluidsefficiëntie per Passagier of Vracht")
    
    # Voeg een slider toe voor capaciteitsaanpassing
    capacity_adjustment = st.slider(
        "Capaciteitsaanpassing (%)",
        min_value=0,
        max_value=100,
        value=100,
        help="Pas de capaciteit aan van 0% (leeg) tot 100% (volle bezetting)"
    )
    
    # Voeg een toggle toe voor analyse type
    analysis_type = st.radio(
        "Analyse type:",
        options=['Geluid per passagier', 'Geluid per ton vracht'],
        horizontal=True
    )
    
    if 'SEL_dB' in df_merged.columns and 'passengers' in df_merged.columns and 'cargo_kg' in df_merged.columns:
        # Pas capaciteit aan op basis van slider
        df_merged['adjusted_passengers'] = df_merged['passengers'] * (capacity_adjustment / 100)
        df_merged['adjusted_cargo_kg'] = df_merged['cargo_kg'] * (capacity_adjustment / 100)
        
        # Bereken SEL per eenheid op basis van geselecteerde analyse type
        if analysis_type == 'Geluid per passagier':
            df_merged['noise_per_unit'] = df_merged['SEL_dB'] / df_merged['adjusted_passengers'].replace(0, np.nan)
            unit_label = "passagier"
            color_scale = 'blues'
        else:
            df_merged['noise_per_unit'] = df_merged['SEL_dB'] / (df_merged['adjusted_cargo_kg'] / 1000).replace(0, np.nan)
            unit_label = "ton vracht"
            color_scale = 'greens'
 
        # Groepeer per vliegtuigtype
        grouped_data = df_merged.groupby('icao_type').agg({
            'noise_per_unit': 'mean',
            'SEL_dB': 'mean',
            'adjusted_passengers': 'first',
            'adjusted_cargo_kg': 'first',
            'Haul_Category': 'first'
        }).reset_index()
        
        # Filter types met minimaal 3 vluchten
        flight_counts = df_merged['icao_type'].value_counts()
        valid_types = flight_counts[flight_counts >= 3].index
        grouped_data = grouped_data[grouped_data['icao_type'].isin(valid_types)].sort_values('noise_per_unit')
 
        # Maak plot
        fig = px.bar(
            grouped_data,
            x='icao_type',
            y='noise_per_unit',
            labels={'icao_type': 'Vliegtuigtype', 'noise_per_unit': f'SEL_dB per {unit_label}'},
            title=f'Geluidsefficiëntie per {unit_label} ({capacity_adjustment}% capaciteit)',
            color='noise_per_unit',
            color_continuous_scale=color_scale,
            hover_data={
                'icao_type': True,
                'noise_per_unit': ':.3f',
                'SEL_dB': ':.1f',
                'adjusted_passengers': ':,.0f',
                'adjusted_cargo_kg': ':,.0f',
                'Haul_Category': True
            }
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
 
        # Toon details tabel
        st.subheader("Details per vliegtuigtype")
        st.dataframe(
            grouped_data.style.format({
                'noise_per_unit': '{:.3f}',
                'SEL_dB': '{:.1f} dB',
                'adjusted_passengers': '{:,.0f}',
                'adjusted_cargo_kg': '{:,.0f} kg'
            }).background_gradient(
                cmap=color_scale.capitalize(),
                subset=['noise_per_unit']
            ).set_properties(
                **{'background-color': '#f0f2f6'},
                subset=['Haul_Category']
            )
        )
        
        # Conclusie
        if not grouped_data.empty:
            best_type = grouped_data.iloc[0]['icao_type']
            best_value = grouped_data.iloc[0]['noise_per_unit']
            st.success(f"**Meest efficiënt:** {best_type} ({grouped_data.iloc[0]['Haul_Category']}) met {best_value:.2f} dB per {unit_label}")
    else:
        st.warning("Benodigde kolommen ontbreken in de data (SEL_dB, passengers of cargo_kg)")
