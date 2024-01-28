use std::fs::File;
use std::io::{Write};
use std::sync::{Arc, Mutex};
use std::thread;
use rand::distributions::{Distribution};
use clap::Parser;
use rand::SeedableRng;
use rand::distributions::{Uniform};
use rand::rngs::{SmallRng, ThreadRng};

struct WeatherStation<'a> {
    city_name: &'a str,
    temperature: f32,
}

const WEATHER_STATIONS: [WeatherStation; 413] = [
    WeatherStation::new("Abha", 18.0),
    WeatherStation::new("Abidjan", 26.0),
    WeatherStation::new("Abéché", 29.4),
    WeatherStation::new("Accra", 26.4),
    WeatherStation::new("Addis Ababa", 16.0),
    WeatherStation::new("Adelaide", 17.3),
    WeatherStation::new("Aden", 29.1),
    WeatherStation::new("Ahvaz", 25.4),
    WeatherStation::new("Albuquerque", 14.0),
    WeatherStation::new("Alexandra", 11.0),
    WeatherStation::new("Alexandria", 20.0),
    WeatherStation::new("Algiers", 18.2),
    WeatherStation::new("Alice Springs", 21.0),
    WeatherStation::new("Almaty", 10.0),
    WeatherStation::new("Amsterdam", 10.2),
    WeatherStation::new("Anadyr", -6.9),
    WeatherStation::new("Anchorage", 2.8),
    WeatherStation::new("Andorra la Vella", 9.8),
    WeatherStation::new("Ankara", 12.0),
    WeatherStation::new("Antananarivo", 17.9),
    WeatherStation::new("Antsiranana", 25.2),
    WeatherStation::new("Arkhangelsk", 1.3),
    WeatherStation::new("Ashgabat", 17.1),
    WeatherStation::new("Asmara", 15.6),
    WeatherStation::new("Assab", 30.5),
    WeatherStation::new("Astana", 3.5),
    WeatherStation::new("Athens", 19.2),
    WeatherStation::new("Atlanta", 17.0),
    WeatherStation::new("Auckland", 15.2),
    WeatherStation::new("Austin", 20.7),
    WeatherStation::new("Baghdad", 22.77),
    WeatherStation::new("Baguio", 19.5),
    WeatherStation::new("Baku", 15.1),
    WeatherStation::new("Baltimore", 13.1),
    WeatherStation::new("Bamako", 27.8),
    WeatherStation::new("Bangkok", 28.6),
    WeatherStation::new("Bangui", 26.0),
    WeatherStation::new("Banjul", 26.0),
    WeatherStation::new("Barcelona", 18.2),
    WeatherStation::new("Bata", 25.1),
    WeatherStation::new("Batumi", 14.0),
    WeatherStation::new("Beijing", 12.9),
    WeatherStation::new("Beirut", 20.9),
    WeatherStation::new("Belgrade", 12.5),
    WeatherStation::new("Belize City", 26.7),
    WeatherStation::new("Benghazi", 19.9),
    WeatherStation::new("Bergen", 7.7),
    WeatherStation::new("Berlin", 10.3),
    WeatherStation::new("Bilbao", 14.7),
    WeatherStation::new("Birao", 26.5),
    WeatherStation::new("Bishkek", 11.3),
    WeatherStation::new("Bissau", 27.0),
    WeatherStation::new("Blantyre", 22.2),
    WeatherStation::new("Bloemfontein", 15.6),
    WeatherStation::new("Boise", 11.4),
    WeatherStation::new("Bordeaux", 14.2),
    WeatherStation::new("Bosaso", 30.0),
    WeatherStation::new("Boston", 10.9),
    WeatherStation::new("Bouaké", 26.0),
    WeatherStation::new("Bratislava", 10.5),
    WeatherStation::new("Brazzaville", 25.0),
    WeatherStation::new("Bridgetown", 27.0),
    WeatherStation::new("Brisbane", 21.4),
    WeatherStation::new("Brussels", 10.5),
    WeatherStation::new("Bucharest", 10.8),
    WeatherStation::new("Budapest", 11.3),
    WeatherStation::new("Bujumbura", 23.8),
    WeatherStation::new("Bulawayo", 18.9),
    WeatherStation::new("Burnie", 13.1),
    WeatherStation::new("Busan", 15.0),
    WeatherStation::new("Cabo San Lucas", 23.9),
    WeatherStation::new("Cairns", 25.0),
    WeatherStation::new("Cairo", 21.4),
    WeatherStation::new("Calgary", 4.4),
    WeatherStation::new("Canberra", 13.1),
    WeatherStation::new("Cape Town", 16.2),
    WeatherStation::new("Changsha", 17.4),
    WeatherStation::new("Charlotte", 16.1),
    WeatherStation::new("Chiang Mai", 25.8),
    WeatherStation::new("Chicago", 9.8),
    WeatherStation::new("Chihuahua", 18.6),
    WeatherStation::new("Chișinău", 10.2),
    WeatherStation::new("Chittagong", 25.9),
    WeatherStation::new("Chongqing", 18.6),
    WeatherStation::new("Christchurch", 12.2),
    WeatherStation::new("City of San Marino", 11.8),
    WeatherStation::new("Colombo", 27.4),
    WeatherStation::new("Columbus", 11.7),
    WeatherStation::new("Conakry", 26.4),
    WeatherStation::new("Copenhagen", 9.1),
    WeatherStation::new("Cotonou", 27.2),
    WeatherStation::new("Cracow", 9.3),
    WeatherStation::new("Da Lat", 17.9),
    WeatherStation::new("Da Nang", 25.8),
    WeatherStation::new("Dakar", 24.0),
    WeatherStation::new("Dallas", 19.0),
    WeatherStation::new("Damascus", 17.0),
    WeatherStation::new("Dampier", 26.4),
    WeatherStation::new("Dar es Salaam", 25.8),
    WeatherStation::new("Darwin", 27.6),
    WeatherStation::new("Denpasar", 23.7),
    WeatherStation::new("Denver", 10.4),
    WeatherStation::new("Detroit", 10.0),
    WeatherStation::new("Dhaka", 25.9),
    WeatherStation::new("Dikson", -11.1),
    WeatherStation::new("Dili", 26.6),
    WeatherStation::new("Djibouti", 29.9),
    WeatherStation::new("Dodoma", 22.7),
    WeatherStation::new("Dolisie", 24.0),
    WeatherStation::new("Douala", 26.7),
    WeatherStation::new("Dubai", 26.9),
    WeatherStation::new("Dublin", 9.8),
    WeatherStation::new("Dunedin", 11.1),
    WeatherStation::new("Durban", 20.6),
    WeatherStation::new("Dushanbe", 14.7),
    WeatherStation::new("Edinburgh", 9.3),
    WeatherStation::new("Edmonton", 4.2),
    WeatherStation::new("El Paso", 18.1),
    WeatherStation::new("Entebbe", 21.0),
    WeatherStation::new("Erbil", 19.5),
    WeatherStation::new("Erzurum", 5.1),
    WeatherStation::new("Fairbanks", -2.3),
    WeatherStation::new("Fianarantsoa", 17.9),
    WeatherStation::new("Flores,  Petén", 26.4),
    WeatherStation::new("Frankfurt", 10.6),
    WeatherStation::new("Fresno", 17.9),
    WeatherStation::new("Fukuoka", 17.0),
    WeatherStation::new("Gabès", 19.5),
    WeatherStation::new("Gaborone", 21.0),
    WeatherStation::new("Gagnoa", 26.0),
    WeatherStation::new("Gangtok", 15.2),
    WeatherStation::new("Garissa", 29.3),
    WeatherStation::new("Garoua", 28.3),
    WeatherStation::new("George Town", 27.9),
    WeatherStation::new("Ghanzi", 21.4),
    WeatherStation::new("Gjoa Haven", -14.4),
    WeatherStation::new("Guadalajara", 20.9),
    WeatherStation::new("Guangzhou", 22.4),
    WeatherStation::new("Guatemala City", 20.4),
    WeatherStation::new("Halifax", 7.5),
    WeatherStation::new("Hamburg", 9.7),
    WeatherStation::new("Hamilton", 13.8),
    WeatherStation::new("Hanga Roa", 20.5),
    WeatherStation::new("Hanoi", 23.6),
    WeatherStation::new("Harare", 18.4),
    WeatherStation::new("Harbin", 5.0),
    WeatherStation::new("Hargeisa", 21.7),
    WeatherStation::new("Hat Yai", 27.0),
    WeatherStation::new("Havana", 25.2),
    WeatherStation::new("Helsinki", 5.9),
    WeatherStation::new("Heraklion", 18.9),
    WeatherStation::new("Hiroshima", 16.3),
    WeatherStation::new("Ho Chi Minh City", 27.4),
    WeatherStation::new("Hobart", 12.7),
    WeatherStation::new("Hong Kong", 23.3),
    WeatherStation::new("Honiara", 26.5),
    WeatherStation::new("Honolulu", 25.4),
    WeatherStation::new("Houston", 20.8),
    WeatherStation::new("Ifrane", 11.4),
    WeatherStation::new("Indianapolis", 11.8),
    WeatherStation::new("Iqaluit", -9.3),
    WeatherStation::new("Irkutsk", 1.0),
    WeatherStation::new("Istanbul", 13.9),
    WeatherStation::new("İzmir", 17.9),
    WeatherStation::new("Jacksonville", 20.3),
    WeatherStation::new("Jakarta", 26.7),
    WeatherStation::new("Jayapura", 27.0),
    WeatherStation::new("Jerusalem", 18.3),
    WeatherStation::new("Johannesburg", 15.5),
    WeatherStation::new("Jos", 22.8),
    WeatherStation::new("Juba", 27.8),
    WeatherStation::new("Kabul", 12.1),
    WeatherStation::new("Kampala", 20.0),
    WeatherStation::new("Kandi", 27.7),
    WeatherStation::new("Kankan", 26.5),
    WeatherStation::new("Kano", 26.4),
    WeatherStation::new("Kansas City", 12.5),
    WeatherStation::new("Karachi", 26.0),
    WeatherStation::new("Karonga", 24.4),
    WeatherStation::new("Kathmandu", 18.3),
    WeatherStation::new("Khartoum", 29.9),
    WeatherStation::new("Kingston", 27.4),
    WeatherStation::new("Kinshasa", 25.3),
    WeatherStation::new("Kolkata", 26.7),
    WeatherStation::new("Kuala Lumpur", 27.3),
    WeatherStation::new("Kumasi", 26.0),
    WeatherStation::new("Kunming", 15.7),
    WeatherStation::new("Kuopio", 3.4),
    WeatherStation::new("Kuwait City", 25.7),
    WeatherStation::new("Kyiv", 8.4),
    WeatherStation::new("Kyoto", 15.8),
    WeatherStation::new("La Ceiba", 26.2),
    WeatherStation::new("La Paz", 23.7),
    WeatherStation::new("Lagos", 26.8),
    WeatherStation::new("Lahore", 24.3),
    WeatherStation::new("Lake Havasu City", 23.7),
    WeatherStation::new("Lake Tekapo", 8.7),
    WeatherStation::new("Las Palmas de Gran Canaria", 21.2),
    WeatherStation::new("Las Vegas", 20.3),
    WeatherStation::new("Launceston", 13.1),
    WeatherStation::new("Lhasa", 7.6),
    WeatherStation::new("Libreville", 25.9),
    WeatherStation::new("Lisbon", 17.5),
    WeatherStation::new("Livingstone", 21.8),
    WeatherStation::new("Ljubljana", 10.9),
    WeatherStation::new("Lodwar", 29.3),
    WeatherStation::new("Lomé", 26.9),
    WeatherStation::new("London", 11.3),
    WeatherStation::new("Los Angeles", 18.6),
    WeatherStation::new("Louisville", 13.9),
    WeatherStation::new("Luanda", 25.8),
    WeatherStation::new("Lubumbashi", 20.8),
    WeatherStation::new("Lusaka", 19.9),
    WeatherStation::new("Luxembourg City", 9.3),
    WeatherStation::new("Lviv", 7.8),
    WeatherStation::new("Lyon", 12.5),
    WeatherStation::new("Madrid", 15.0),
    WeatherStation::new("Mahajanga", 26.3),
    WeatherStation::new("Makassar", 26.7),
    WeatherStation::new("Makurdi", 26.0),
    WeatherStation::new("Malabo", 26.3),
    WeatherStation::new("Malé", 28.0),
    WeatherStation::new("Managua", 27.3),
    WeatherStation::new("Manama", 26.5),
    WeatherStation::new("Mandalay", 28.0),
    WeatherStation::new("Mango", 28.1),
    WeatherStation::new("Manila", 28.4),
    WeatherStation::new("Maputo", 22.8),
    WeatherStation::new("Marrakesh", 19.6),
    WeatherStation::new("Marseille", 15.8),
    WeatherStation::new("Maun", 22.4),
    WeatherStation::new("Medan", 26.5),
    WeatherStation::new("Mek'ele", 22.7),
    WeatherStation::new("Melbourne", 15.1),
    WeatherStation::new("Memphis", 17.2),
    WeatherStation::new("Mexicali", 23.1),
    WeatherStation::new("Mexico City", 17.5),
    WeatherStation::new("Miami", 24.9),
    WeatherStation::new("Milan", 13.0),
    WeatherStation::new("Milwaukee", 8.9),
    WeatherStation::new("Minneapolis", 7.8),
    WeatherStation::new("Minsk", 6.7),
    WeatherStation::new("Mogadishu", 27.1),
    WeatherStation::new("Mombasa", 26.3),
    WeatherStation::new("Monaco", 16.4),
    WeatherStation::new("Moncton", 6.1),
    WeatherStation::new("Monterrey", 22.3),
    WeatherStation::new("Montreal", 6.8),
    WeatherStation::new("Moscow", 5.8),
    WeatherStation::new("Mumbai", 27.1),
    WeatherStation::new("Murmansk", 0.6),
    WeatherStation::new("Muscat", 28.0),
    WeatherStation::new("Mzuzu", 17.7),
    WeatherStation::new("N'Djamena", 28.3),
    WeatherStation::new("Naha", 23.1),
    WeatherStation::new("Nairobi", 17.8),
    WeatherStation::new("Nakhon Ratchasima", 27.3),
    WeatherStation::new("Napier", 14.6),
    WeatherStation::new("Napoli", 15.9),
    WeatherStation::new("Nashville", 15.4),
    WeatherStation::new("Nassau", 24.6),
    WeatherStation::new("Ndola", 20.3),
    WeatherStation::new("New Delhi", 25.0),
    WeatherStation::new("New Orleans", 20.7),
    WeatherStation::new("New York City", 12.9),
    WeatherStation::new("Ngaoundéré", 22.0),
    WeatherStation::new("Niamey", 29.3),
    WeatherStation::new("Nicosia", 19.7),
    WeatherStation::new("Niigata", 13.9),
    WeatherStation::new("Nouadhibou", 21.3),
    WeatherStation::new("Nouakchott", 25.7),
    WeatherStation::new("Novosibirsk", 1.7),
    WeatherStation::new("Nuuk", -1.4),
    WeatherStation::new("Odesa", 10.7),
    WeatherStation::new("Odienné", 26.0),
    WeatherStation::new("Oklahoma City", 15.9),
    WeatherStation::new("Omaha", 10.6),
    WeatherStation::new("Oranjestad", 28.1),
    WeatherStation::new("Oslo", 5.7),
    WeatherStation::new("Ottawa", 6.6),
    WeatherStation::new("Ouagadougou", 28.3),
    WeatherStation::new("Ouahigouya", 28.6),
    WeatherStation::new("Ouarzazate", 18.9),
    WeatherStation::new("Oulu", 2.7),
    WeatherStation::new("Palembang", 27.3),
    WeatherStation::new("Palermo", 18.5),
    WeatherStation::new("Palm Springs", 24.5),
    WeatherStation::new("Palmerston North", 13.2),
    WeatherStation::new("Panama City", 28.0),
    WeatherStation::new("Parakou", 26.8),
    WeatherStation::new("Paris", 12.3),
    WeatherStation::new("Perth", 18.7),
    WeatherStation::new("Petropavlovsk-Kamchatsky", 1.9),
    WeatherStation::new("Philadelphia", 13.2),
    WeatherStation::new("Phnom Penh", 28.3),
    WeatherStation::new("Phoenix", 23.9),
    WeatherStation::new("Pittsburgh", 10.8),
    WeatherStation::new("Podgorica", 15.3),
    WeatherStation::new("Pointe-Noire", 26.1),
    WeatherStation::new("Pontianak", 27.7),
    WeatherStation::new("Port Moresby", 26.9),
    WeatherStation::new("Port Sudan", 28.4),
    WeatherStation::new("Port Vila", 24.3),
    WeatherStation::new("Port-Gentil", 26.0),
    WeatherStation::new("Portland (OR)", 12.4),
    WeatherStation::new("Porto", 15.7),
    WeatherStation::new("Prague", 8.4),
    WeatherStation::new("Praia", 24.4),
    WeatherStation::new("Pretoria", 18.2),
    WeatherStation::new("Pyongyang", 10.8),
    WeatherStation::new("Rabat", 17.2),
    WeatherStation::new("Rangpur", 24.4),
    WeatherStation::new("Reggane", 28.3),
    WeatherStation::new("Reykjavík", 4.3),
    WeatherStation::new("Riga", 6.2),
    WeatherStation::new("Riyadh", 26.0),
    WeatherStation::new("Rome", 15.2),
    WeatherStation::new("Roseau", 26.2),
    WeatherStation::new("Rostov-on-Don", 9.9),
    WeatherStation::new("Sacramento", 16.3),
    WeatherStation::new("Saint Petersburg", 5.8),
    WeatherStation::new("Saint-Pierre", 5.7),
    WeatherStation::new("Salt Lake City", 11.6),
    WeatherStation::new("San Antonio", 20.8),
    WeatherStation::new("San Diego", 17.8),
    WeatherStation::new("San Francisco", 14.6),
    WeatherStation::new("San Jose", 16.4),
    WeatherStation::new("San José", 22.6),
    WeatherStation::new("San Juan", 27.2),
    WeatherStation::new("San Salvador", 23.1),
    WeatherStation::new("Sana'a", 20.0),
    WeatherStation::new("Santo Domingo", 25.9),
    WeatherStation::new("Sapporo", 8.9),
    WeatherStation::new("Sarajevo", 10.1),
    WeatherStation::new("Saskatoon", 3.3),
    WeatherStation::new("Seattle", 11.3),
    WeatherStation::new("Ségou", 28.0),
    WeatherStation::new("Seoul", 12.5),
    WeatherStation::new("Seville", 19.2),
    WeatherStation::new("Shanghai", 16.7),
    WeatherStation::new("Singapore", 27.0),
    WeatherStation::new("Skopje", 12.4),
    WeatherStation::new("Sochi", 14.2),
    WeatherStation::new("Sofia", 10.6),
    WeatherStation::new("Sokoto", 28.0),
    WeatherStation::new("Split", 16.1),
    WeatherStation::new("St. John's", 5.0),
    WeatherStation::new("St. Louis", 13.9),
    WeatherStation::new("Stockholm", 6.6),
    WeatherStation::new("Surabaya", 27.1),
    WeatherStation::new("Suva", 25.6),
    WeatherStation::new("Suwałki", 7.2),
    WeatherStation::new("Sydney", 17.7),
    WeatherStation::new("Tabora", 23.0),
    WeatherStation::new("Tabriz", 12.6),
    WeatherStation::new("Taipei", 23.0),
    WeatherStation::new("Tallinn", 6.4),
    WeatherStation::new("Tamale", 27.9),
    WeatherStation::new("Tamanrasset", 21.7),
    WeatherStation::new("Tampa", 22.9),
    WeatherStation::new("Tashkent", 14.8),
    WeatherStation::new("Tauranga", 14.8),
    WeatherStation::new("Tbilisi", 12.9),
    WeatherStation::new("Tegucigalpa", 21.7),
    WeatherStation::new("Tehran", 17.0),
    WeatherStation::new("Tel Aviv", 20.0),
    WeatherStation::new("Thessaloniki", 16.0),
    WeatherStation::new("Thiès", 24.0),
    WeatherStation::new("Tijuana", 17.8),
    WeatherStation::new("Timbuktu", 28.0),
    WeatherStation::new("Tirana", 15.2),
    WeatherStation::new("Toamasina", 23.4),
    WeatherStation::new("Tokyo", 15.4),
    WeatherStation::new("Toliara", 24.1),
    WeatherStation::new("Toluca", 12.4),
    WeatherStation::new("Toronto", 9.4),
    WeatherStation::new("Tripoli", 20.0),
    WeatherStation::new("Tromsø", 2.9),
    WeatherStation::new("Tucson", 20.9),
    WeatherStation::new("Tunis", 18.4),
    WeatherStation::new("Ulaanbaatar", -0.4),
    WeatherStation::new("Upington", 20.4),
    WeatherStation::new("Ürümqi", 7.4),
    WeatherStation::new("Vaduz", 10.1),
    WeatherStation::new("Valencia", 18.3),
    WeatherStation::new("Valletta", 18.8),
    WeatherStation::new("Vancouver", 10.4),
    WeatherStation::new("Veracruz", 25.4),
    WeatherStation::new("Vienna", 10.4),
    WeatherStation::new("Vientiane", 25.9),
    WeatherStation::new("Villahermosa", 27.1),
    WeatherStation::new("Vilnius", 6.0),
    WeatherStation::new("Virginia Beach", 15.8),
    WeatherStation::new("Vladivostok", 4.9),
    WeatherStation::new("Warsaw", 8.5),
    WeatherStation::new("Washington, D.C.", 14.6),
    WeatherStation::new("Wau", 27.8),
    WeatherStation::new("Wellington", 12.9),
    WeatherStation::new("Whitehorse", -0.1),
    WeatherStation::new("Wichita", 13.9),
    WeatherStation::new("Willemstad", 28.0),
    WeatherStation::new("Winnipeg", 3.0),
    WeatherStation::new("Wrocław", 9.6),
    WeatherStation::new("Xi'an", 14.1),
    WeatherStation::new("Yakutsk", -8.8),
    WeatherStation::new("Yangon", 27.5),
    WeatherStation::new("Yaoundé", 23.8),
    WeatherStation::new("Yellowknife", -4.3),
    WeatherStation::new("Yerevan", 12.4),
    WeatherStation::new("Yinchuan", 9.0),
    WeatherStation::new("Zagreb", 10.7),
    WeatherStation::new("Zanzibar City", 26.0),
    WeatherStation::new("Zürich", 9.3)
];

impl<'a> WeatherStation<'a> {
    const fn new(city_name: &str, temperature: f32) -> WeatherStation {
        WeatherStation {
            city_name,
            temperature,
        }
    }

    fn write_into_buffer(&self, random_offset: f32, buffer: &mut Vec<u8>) {
        buffer.extend_from_slice(self.city_name.as_bytes());
        buffer.push(b';');
        let temperature = self.formatted_str(self.temperature + random_offset);
        buffer.extend_from_slice(temperature.to_string().as_bytes());
        buffer.push(b'\n');
    }
    #[inline]
    fn formatted_str(&self, temperature: f32) -> String {
        let temperature = Self::round(temperature);
        format!("{:.1}", temperature)
    }
    #[inline]
    fn round(f: f32) -> f32 {
        (f * 10f32).round() / 10f32
    }
}


#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Command {
    num_cities: usize,
}

fn write_to_file(batch_size: usize, total_size: usize, file: Arc<Mutex<File>>) {
    let weather_stations = WEATHER_STATIONS;
    let mut sampled = 0;

    let mut buffer: Vec<u8> = Vec::with_capacity(batch_size * 128);
    let thread_rng = ThreadRng::default();

    let weather_station_uniform = Uniform::new(0, weather_stations.len());
    let mut weather_station_rng = SmallRng::from_rng(thread_rng.clone()).unwrap();
    let mut weather_station_sampler = weather_station_uniform.sample_iter(&mut weather_station_rng);

    let uniform = Uniform::new(-10.0, 10.0);
    let mut temperature_rng = SmallRng::from_rng(thread_rng.clone()).unwrap();
    let mut temperature_sampler = uniform.sample_iter(&mut temperature_rng);

    while sampled < total_size {
        let mut buffered = 0;
        let size = batch_size.min(total_size - sampled);
        while buffered < size {
            let weather_station_index_1 = weather_station_sampler.next().unwrap();
            let random_temperature_offset_1 = temperature_sampler.next().unwrap();

            let weather_station_1 = &weather_stations[weather_station_index_1];
            weather_station_1.write_into_buffer(random_temperature_offset_1, &mut buffer);
            buffered+=1;
        }

        sampled += batch_size;
        let read = file.lock().unwrap().write(buffer.as_slice()).unwrap();
        assert_eq!(read, buffer.len());
        buffer.clear();
    }
}


fn main() {

    let Command{num_cities} = Command::parse();
    let path = "measurements.txt";
    let file = Arc::new(Mutex::new(File::create(path).unwrap()));

    thread::scope(move |scope| {
        let num_threads = 10;
        let mut threads = vec![];
        for _ in 0..num_threads {
            let cloned_file =  file.clone();
            let size = num_cities / num_threads;
            let thread =  scope.spawn(move || {
                write_to_file(2<<16, size, cloned_file);
            });
            threads.push(thread);
        }
        for thread in threads {
            thread.join().unwrap();
        }
        file.lock().unwrap().flush().unwrap();
    });

}

#[test]
fn rounding_formatted_test() {

    let station = WeatherStation::new("foo", 0.9);
    assert_eq!("0.9", station.formatted_str(station.temperature));

    let station = WeatherStation::new("foo", -1.142323);
    assert_eq!("-1.1", station.formatted_str(station.temperature));

    let station = WeatherStation::new("foo", -11.0);
    assert_eq!("-11.0", station.formatted_str(station.temperature));

}